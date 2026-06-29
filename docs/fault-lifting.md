# Fault lifting

This document describes *fault lifting*: a technique for evaluating the effect
of a hardware register fault in a systolic array without simulating the array
itself. Instead of running the fault through a cycle-accurate model, we
translate ("lift") the fault into a small set of operations on the input and
output matrices of the matrix multiplication. The result is bit-exact for
integer arithmetic and dramatically faster, especially for batched inputs (See
[Benchmarks](#benchmarks)).

The document starts from the underlying hardware model and builds up to the
lifting algorithms in stages. No prior familiarity with the codebase is assumed.

## Background

### Weight-stationary systolic arrays

A systolic array is a grid of *processing elements* (PEs) used to compute matrix
multiplications. We use the *weight-stationary* style: the weights are loaded
into the grid and stay put, while activations stream through.

We compute

```
output = weights · activations
```

where (with `in_features` the shared/contracted dimension):

- `weights` has shape `(out_features, in_features)`,
- `activations` has shape `(in_features, batch)`,
- `output` has shape `(out_features, batch)`.

The array is a grid of PEs with `nrows` rows (the `y` axis) and `ncols` columns
(the `x` axis). Each PE holds three registers:

- a **weight** register (a stationary weight value),
- an **activation** register (the activation value currently passing through),
- an **accumulator** register (a running partial sum).

The data flows in three directions:

- **Activations flow left to right.** An activation enters a row on the left and
  is handed from each PE to its right neighbour.
- **Partial sums flow top to bottom.** Each PE multiplies its activation by its
  weight, adds the partial sum coming from the PE above, and passes the result
  down.
- **Weights are loaded top to bottom.** Each weight is shifted down through the
  PEs above its final position before settling into place. This detail matters
  for weight faults (see below).

The output leaves from the bottom row of accumulators, but not all at once. The
array is a pipeline: a value entering the top-left takes time to ripple to the
bottom-right, and each batch item follows the one before it. To keep the right
activation meeting the right weight at the right moment, activations are fed in
**zero-padded and staggered**, and the outputs emerge **staggered** in turn,
each column of the bottom row completing on a later cycle than the column to its
left.

The diagram below sketches the data flow for our 2x2 example with a single input
vector `[a0, a1]`. Activations enter from the left; the upper row leads and lower
rows are delayed by a zero, so that each top-row partial sum arrives at the row
below exactly as that row's own activation does. Finished outputs then drop out
of the bottom one column per cycle:

```
   feed in (earliest on the left, zero-padded)
                   *------*------*
   row 0: a0  0 ─> | (0,0)| (0,1)|
                   *------*------*
   row 1:  0 a1 ─> | (1,0)| (1,1)|
                   *------*------*
                      │      │
                      v      v
                  output 0  output 1
                  cycle t   cycle t+1
```

So "read the output" really means "collect each bottom-row accumulator on the
cycle it finishes". This staggering is a property of the physical schedule; the
lifting technique below works entirely in terms of the finished matrices and
does not have to reproduce the timing.

The array stores the weight matrix *transposed*. The reason is the direction of
accumulation: the multiplication contracts over the `in_features` dimension, and
partial sums accumulate *down a column*, so the `in_features` axis must run along
the array's rows. Each column then accumulates one full dot product and yields a
single output, so the `out_features` axis runs along the columns. The PE at array
position `(y, x)` therefore holds `weights[x, y]` — activation row `y` (an
`in_features` index) meeting output row `x` (an `out_features` index). Laying
`in_features` along the rows is also what lets the activations stream in their
natural orientation without reshaping. In short:

- **array row `y` ↔ an activation row** (a row of `activations`, i.e. one
  `in_features` index),
- **array column `x` ↔ an output row** (a row of `output`, i.e. one
  `out_features` index).

Keep this correspondence in mind; every lifting rule is just a restatement of a
physical fault in terms of these row/column identities.

### Mappings and passes

A matrix multiplication can be larger than the physical array, so it is split
into **passes**, each of which fits in the array. A `Mapping` is an ordered list
of passes. Each pass declares:

- which **activation rows** it feeds in, mapped onto a band of array rows,
- which **output rows** it produces, mapped onto a band of array columns,
- optional offsets, so a pass need not start at the top-left corner of the
  array.

(The bands are contiguous ranges rather than arbitrary sets of rows/columns.
This is a simplification of the implementation, not something the lifting
technique relies on; any one-to-one assignment of data rows to array rows would
work the same way.)

Output rows may appear in more than one pass; their partial results are summed.
This is how an output that needs more `in_features` than the array has rows is
built up: each pass contributes part of the dot product.

A mapping is **valid** when, taken together, its passes actually carry out the
full matrix multiplication. The useful way to picture this is in terms of
*connections*: a pass connects every activation row it feeds in to every output
row it produces, because within that pass the activation flows across the array
and contributes to each of those outputs. For the whole computation to be the
correct product, **every (activation row, output row) pair must be connected
exactly once** across all passes:

- *at least once*, or the term `weights[o, i] * activations[i]` would be missing
  from output `o`;
- *at most once*, or that term would be counted twice (for example, two
  overlapping passes that both connect the same pair).

On top of this coverage requirement, two consistency invariants are what make
lifting possible:

1. **Row consistency.** A given activation row is always mapped to the *same*
   array row, in every pass it appears in.
2. **Column consistency.** A given output row is always mapped to the *same*
   array column, in every pass it appears in.

These invariants are what let us aggregate a fault's effect across all passes
with a single corrected matmul, as explained in the final section.

### The fault model

We consider **single stuck-at faults** in one register of one PE. Such a fault
is fully described by:

- the target PE position `(y, x)`,
- which register is affected (weight, activation, or accumulator),
- a bit index,
- a stuck value (stuck-at-zero or stuck-at-one).

Applying the fault to a value means forcing that one bit to the stuck value;
every other bit is untouched. We write `corrupt(v)` for "the value `v` with the
stuck bit forced". `corrupt` is idempotent (applying it twice changes nothing),
which is why it does not matter whether the hardware corrupts a value when it is
written or when it is read.

## The idea of lifting

A fault lives in **array space**: it is a property of a specific register in a
specific PE. But the only thing an experiment ultimately cares about is the
*output* of the multiplication. Lifting rewrites the fault into **matrix space**:
a recipe expressed purely in terms of the `weights`, `activations`, and `output`
matrices.

Why bother? Simulating the array faithfully means stepping it cycle by cycle,
and the cost grows with the number of PEs and the number of cycles, with most of
the work being inherently sequential and repeated for every item in a batch.
Lifting collapses the same result into one or two dense matrix multiplications
(which can use optimized BLAS routines and are amortized over the whole batch)
plus a handful of cheap, sparse fix-ups.

There is exactly one lifting strategy per register kind. We describe each in
turn, first physically, then as a matrix-space recipe, with a tiny worked
example using a single activation vector (batch size 1). Batching is handled
afterwards, once the single-vector picture is clear.

For all examples we use a 2x2 array computing a 2x2-by-2x1 product:

```
weights = | w00  w01 |      activations = | a0 |
          | w10  w11 |                    | a1 |

output  = | w00*a0 + w01*a1 |   (output row 0)
          | w10*a0 + w11*a1 |   (output row 1)
```

A single pass maps activation rows {0, 1} onto array rows {0, 1} and output rows
{0, 1} onto array columns {0, 1}. The weight held at array position `(y, x)` is
`weights[x, y]` (note the transpose):

```
array(0,0) = w00     array(0,1) = w10
array(1,0) = w01     array(1,1) = w11
```

## Case 1: weight register faults

### Physical behaviour

Weights load from the top and shift down into place. Every weight destined for a
row at or below the faulty PE must pass *through* the faulty register on its way
down, so it picks up the stuck bit. Weights destined for rows *above* the faulty
PE never touch it.

So a weight fault at `(y0, x0)` corrupts the stored weights in column `x0` for
every array row `y >= y0`.

### Lifted recipe

Translate "column `x0`, rows `y >= y0`" into matrix space:

- column `x0` is a single output row `o`;
- rows `y >= y0` are the activation rows mapped to those array rows.

The fault therefore corrupts a specific set of entries of the weight matrix:
`weights[o, r]` for each affected activation row `r`. Once those entries are
corrupted, the computation is just an ordinary multiplication with a modified
weight matrix.

> **Recipe.** Corrupt the affected weight-matrix entries in place, then do a
> normal matmul.

This case is the simplest because a stuck weight is, by definition, a change to
the weight matrix and nothing else.

### Example

Fault at `array(1, 0)`, weight register. Column `x0 = 0` is output row 0; array
rows `y >= 1` cover activation row 1. The affected entry is `weights[0, 1] =
w01`. The lifted output is:

```
output row 0 = w00*a0 + corrupt(w01)*a1     (changed)
output row 1 = w10*a0 + w11*a1              (unchanged)
```

## Case 2: activation register faults

### Physical behaviour

An activation streams left to right along its row. When it reaches the faulty
PE at `(y0, x0)` it is corrupted, and the corrupted value continues rightward to
every PE further along the row. PEs to the *left* of `x0` saw the clean value.

So an activation fault at `(y0, x0)`:

- corrupts the activation in the row mapped to array row `y0`,
- but only as seen by columns `x >= x0`, i.e. only the output rows produced by
  those columns.

### Lifted recipe

This is a **row substitution**. The affected activation row is corrupted
*entirely* (in a copy of the activations), and the multiplication is run a
second time. The final output takes the affected output rows from this faulty
run and every other output row from the clean run.

> **Recipe.**
> 1. Compute the clean output.
> 2. Make a copy of the activations with the affected activation row(s)
>    corrupted, and compute a second, faulty output.
> 3. For each affected output row, replace the clean row with the faulty row.

Why is it correct to corrupt the *whole* activation row and then only keep some
output rows? An output row to the *left* of `x0` saw the clean activation, so we
keep its clean value. An output row at or to the right of `x0` saw the corrupted
activation, and its faulty value already reflects exactly that. Corrupting the
whole row in the faulty pass is harmless because we discard the rows we should
not have changed.

### Example

Fault at `array(0, 1)`, activation register. The corrupted activation row is the
one mapped to array row 0, namely activation row 0 (value `a0`). Columns
`x >= 1` produce output row 1, so only output row 1 is affected:

```
output row 0 = w00*a0          + w01*a1   (clean, column 0 saw clean a0)
output row 1 = w10*corrupt(a0) + w11*a1   (faulty, column 1 saw corrupted a0)
```

## Case 3: accumulator register faults

### Physical behaviour

The accumulator at `(y0, x0)` holds the partial sum *after* this PE has added its
own product to the partial sum arriving from above. A stuck bit corrupts that
partial sum, and the corrupted value flows down the column, with the PEs below
adding their (clean) products on top of it. The bottom of column `x0` is a single
output row, so an accumulator fault affects exactly one output row per pass.

The partial sum held at `(y0, x0)` is the sum of the products from the used rows
*at and above* `y0`. Everything below `y0` is added cleanly afterwards.

### Lifted recipe

Let `o` be the output row produced by column `x0`, and let `P` be the partial
sum accumulated through row `y0` (the contribution of the activation rows whose
products have reached the faulty PE). The clean output for `o` decomposes as

```
clean(o) = P + (rest of the sum)
```

and the hardware produces

```
faulty(o) = corrupt(P) + (rest of the sum).
```

Subtracting gives a fix-up that needs only the partial sum:

> **Recipe.** For the affected output row, compute the partial sum `P` over the
> contributing activation rows, then apply
>
> ```
> faulty(o) = clean(o) - P + corrupt(P).
> ```

`P` is itself just a small matrix product: the affected output row's weights,
restricted to the contributing activation columns, times those activation rows.

### One recipe, every position

The only thing that varies between accumulator faults is *which* activation rows
make up `P`, and that follows from where the faulty PE sits relative to the rows
a given pass actually uses:

- **within** the used rows - `P` is the partial sum down to and including the
  fault;
- **below** the used rows - the whole column has already accumulated, so `P` is
  the *full* column sum;
- **above** the used rows - nothing has accumulated, so `P` is empty and equals
  zero.

The last case is worth noting in its own right: a stuck bit in a PE the pass does
not even use still matters, because that PE's accumulator (holding zero) is
corrupted and the corrupted value flows down the column. A stuck-at-one there
adds the constant `corrupt(0)` to every output in the column; a stuck-at-zero
does nothing. It is not a separate rule, just the recipe with `P = 0` - an empty
partial sum is still a partial sum.

### Example

Fault at `array(1, 0)`, accumulator register. Column 0 produces output row 0.
The contributing rows are array rows 0 and 1, i.e. activation rows {0, 1}, so the
partial sum is the entire column-0 sum:

```
P        = w00*a0 + w01*a1
clean(0) = P
faulty(0)= clean(0) - P + corrupt(P) = corrupt(w00*a0 + w01*a1)
```

which matches the hardware: the bottom accumulator of column 0 holds the full
sum and is corrupted just before it leaves the array.

For the faulty-zero variant, imagine a taller array where a pass uses only the
lower rows and the fault sits in an unused upper row of some column. Every output
produced by that column gains a constant `corrupt(0)` (for stuck-at-one bit `k`,
that constant is `2^k`).

## From a single vector to batches

The examples above used a single activation vector. In practice `activations`
has many columns (a batch), and this is where lifting pays off, because the work
barely grows with the batch.

- **Weight faults.** The corrupted entries of the weight matrix are independent
  of the batch. A single matmul with the modified weights handles all batch
  items at once.
- **Activation faults.** Corrupting an activation *row* corrupts that input for
  every batch item simultaneously (it is a full row of the activation matrix).
  The whole case costs two dense matmuls regardless of batch size, plus a cheap
  per-row copy of the affected output rows.
- **Accumulator faults.** The partial sum `P` becomes a *vector* over the batch
  (one partial sum per batch item), obtained from a single small matrix product.
  The fix-up `clean - P + corrupt(P)` is then applied element-wise across the
  batch for the affected output row.

In every case the expensive part is one or two dense matrix multiplications,
which optimized linear-algebra libraries make very fast and which amortize over
the entire batch. Compare this with array simulation, where the per-cycle work
must be repeated for each batch item. The larger the batch, the bigger the win.

## Why aggregation across passes is valid

A fault must be lifted with respect to *all* passes at once, because a single
array position participates in many passes. The lifted recipes simply union the
affected rows/columns over every pass and then run the corrected matmul once. The
mapping consistency invariants are what make this sound.

Consider an activation fault. Across passes, array row `y0` may map to several
different activation rows, and we corrupt all of them in one faulty matmul; we
also collect every affected output row. For this single combined pass to be
correct, we need: *for every affected output row `o` and every corrupted
activation row `r`, the contribution of `r` to `o` really should be corrupted.*

Row consistency guarantees that if `r` was corrupted then `r` maps to array row
`y0` in *every* pass containing it. Column consistency guarantees that if `o` was
affected then `o` maps to a column `x >= x0` in *every* pass containing it. The
single connection between `r` and `o` lives in exactly one pass, and in that pass
`r` sits at row `y0` and `o` sits at a column `x >= x0`, which is precisely the
condition under which the hardware corrupts that contribution. So the combined
faulty matmul corrupts exactly the right contributions and no others.

The same reasoning underpins the weight and accumulator cases: because each
activation row and each output row has a fixed home in the array, a fault's
footprint can be described once, in matrix space, and applied with a single
corrected computation.

## Summary

| Register     | What the fault touches                             | Lifted operation                                                          |
|--------------|----------------------------------------------------|---------------------------------------------------------------------------|
| Weight       | Weights in one column, at/below the PE row         | Corrupt those weight-matrix entries; one matmul                           |
| Activation   | One activation row, for columns at/right of the PE | Corrupt the activation row(s); second matmul; splice affected output rows |
| Accumulator  | One output row's partial sum (possibly empty)      | `output = clean - partial + corrupt(partial)` per affected output row     |

In all cases the lifted fault produces bit-exact integer results identical to
running the fault through the array, at a fraction of the cost.

## Benchmarks

The benchmarks below were produced by `crates/systolic/benches/fault_lifting.rs`
using the `divan` crate. Each `literal` variant simulates the fault cycle-by-cycle
through the array; each `lifted` variant uses the recipes above. Both paths use
`f32` matrices. The lifted path uses ndarray's built-in matrix multiplication
(`matrixmultiply`); no BLAS acceleration is enabled.

### Key takeaways

- **The speedup scales with array size.** A 64×64 array requires simulating 4096
  PEs per multiplication. At that size, lifted is 600-1600× faster at small batch
  sizes for weight and accumulator faults.
- **The speedup is consistent across batch sizes.** Literal simulation repeats
  per-PE work for every batch item. Lifted amortises the same dense matmul over
  the entire batch. The ratio does narrow at large batches (both sides grow
  proportionally), but lifted is never slower.
- **Multiple passes multiply the advantage.** When the weight matrix is larger
  than the physical array, the literal path must simulate each pass separately.
  The multi-pass benchmark (8×8 array, 64×64 weights, 64 passes) shows lifted
  running a single 64×64 matmul where literal chains 64 separate array simulations
  - giving a ~56-72× speedup even on a small array.
- **The floor for the lifted path is one dense matmul.** Even for trivial
  configurations (8×8, batch 1), lifted takes only a few hundred nanoseconds
  because the dominant work is a well-optimised matrix multiplication. Literal
  can never be faster than its per-PE simulation loop.

### Full results

```
fault_lifting               fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ accumulator_fault                      │               │               │               │         │
│  ├─ lifted                              │               │               │               │         │
│  │  ├─ arr8x8_batch1      226.6 ns      │ 9.612 µs      │ 239.6 ns      │ 338 ns        │ 100     │ 100
│  │  ├─ arr8x8_batch64     834.6 ns      │ 2.907 µs      │ 869.1 ns      │ 903.5 ns      │ 100     │ 100
│  │  ├─ arr8x8_batch256    2.697 µs      │ 9.506 µs      │ 2.721 µs      │ 2.797 µs      │ 100     │ 100
│  │  ├─ arr64x64_batch1    3.694 µs      │ 48.53 µs      │ 3.79 µs       │ 4.238 µs      │ 100     │ 100
│  │  ├─ arr64x64_batch64   21.26 µs      │ 45.58 µs      │ 21.48 µs      │ 21.94 µs      │ 100     │ 100
│  │  ╰─ arr64x64_batch256  48.13 µs      │ 161.6 µs      │ 53.49 µs      │ 58.62 µs      │ 100     │ 100
│  ╰─ literal                             │               │               │               │         │
│     ├─ arr8x8_batch1      4.46 µs       │ 9.922 µs      │ 4.585 µs      │ 4.694 µs      │ 100     │ 100
│     ├─ arr8x8_batch64     20.62 µs      │ 50.38 µs      │ 23.81 µs      │ 24.54 µs      │ 100     │ 100
│     ├─ arr8x8_batch256    79.24 µs      │ 100.6 µs      │ 80.09 µs      │ 81 µs         │ 100     │ 100
│     ├─ arr64x64_batch1    2.345 ms      │ 4.127 ms      │ 2.529 ms      │ 2.68 ms       │ 100     │ 100
│     ├─ arr64x64_batch64   3.912 ms      │ 7.21 ms       │ 4.15 ms       │ 4.67 ms       │ 100     │ 100
│     ╰─ arr64x64_batch256  8.627 ms      │ 15.38 ms      │ 9.81 ms       │ 10.21 ms      │ 100     │ 100
├─ activation_fault                       │               │               │               │         │
│  ├─ lifted                              │               │               │               │         │
│  │  ├─ arr8x8_batch1      295.6 ns      │ 3.733 µs      │ 305.6 ns      │ 380 ns        │ 100     │ 100
│  │  ├─ arr8x8_batch64     906.6 ns      │ 3.115 µs      │ 959.6 ns      │ 1.004 µs      │ 100     │ 100
│  │  ├─ arr8x8_batch256    2.574 µs      │ 4.829 µs      │ 2.623 µs      │ 2.651 µs      │ 100     │ 100
│  │  ├─ arr64x64_batch1    4.624 µs      │ 9.186 µs      │ 4.739 µs      │ 4.908 µs      │ 100     │ 100
│  │  ├─ arr64x64_batch64   20.06 µs      │ 55.48 µs      │ 20.27 µs      │ 20.91 µs      │ 100     │ 100
│  │  ╰─ arr64x64_batch256  130.8 µs      │ 201.6 µs      │ 138 µs        │ 140.4 µs      │ 100     │ 100
│  ╰─ literal                             │               │               │               │         │
│     ├─ arr8x8_batch1      8.045 µs      │ 20.83 µs      │ 8.163 µs      │ 8.351 µs      │ 100     │ 100
│     ├─ arr8x8_batch64     28.22 µs      │ 41.86 µs      │ 31.94 µs      │ 31.8 µs       │ 100     │ 100
│     ├─ arr8x8_batch256    100.2 µs      │ 116.3 µs      │ 101 µs        │ 101.9 µs      │ 100     │ 100
│     ├─ arr64x64_batch1    2.879 ms      │ 3.821 ms      │ 2.997 ms      │ 3.022 ms      │ 100     │ 100
│     ├─ arr64x64_batch64   4.037 ms      │ 4.518 ms      │ 4.207 ms      │ 4.208 ms      │ 100     │ 100
│     ╰─ arr64x64_batch256  7.229 ms      │ 12.05 ms      │ 7.678 ms      │ 7.748 ms      │ 100     │ 100
├─ multi_pass                             │               │               │               │         │
│  ├─ lifted                              │               │               │               │         │
│  │  ├─ arr8x8_batch1      5.391 µs      │ 12.36 µs      │ 6.013 µs      │ 6.164 µs      │ 100     │ 100
│  │  ├─ arr8x8_batch64     21.89 µs      │ 29.63 µs      │ 22.06 µs      │ 22.23 µs      │ 100     │ 100
│  │  ╰─ arr8x8_batch256    90.75 µs      │ 124.4 µs      │ 94.64 µs      │ 96.29 µs      │ 100     │ 100
│  ╰─ literal                             │               │               │               │         │
│     ├─ arr8x8_batch1      293.5 µs      │ 364.4 µs      │ 336.3 µs      │ 338 µs        │ 100     │ 100
│     ├─ arr8x8_batch64     1.567 ms      │ 2.322 ms      │ 1.59 ms       │ 1.655 ms      │ 100     │ 100
│     ╰─ arr8x8_batch256    5.321 ms      │ 22.76 ms      │ 6.772 ms      │ 8.694 ms      │ 100     │ 100
╰─ weight_fault                           │               │               │               │         │
   ├─ lifted                              │               │               │               │         │
   │  ├─ arr8x8_batch1      223.6 ns      │ 5.223 µs      │ 308.1 ns      │ 358.9 ns      │ 100     │ 100
   │  ├─ arr8x8_batch64     637.1 ns      │ 9.908 µs      │ 788.8 ns      │ 889.3 ns      │ 100     │ 200
   │  ├─ arr8x8_batch256    1.82 µs       │ 3.656 µs      │ 2.141 µs      │ 2.17 µs       │ 100     │ 100
   │  ├─ arr64x64_batch1    3.532 µs      │ 5.122 µs      │ 4.386 µs      │ 4.331 µs      │ 100     │ 100
   │  ├─ arr64x64_batch64   14.15 µs      │ 232.2 µs      │ 17.76 µs      │ 20.47 µs      │ 100     │ 100
   │  ╰─ arr64x64_batch256  89.64 µs      │ 173.8 µs      │ 103.6 µs      │ 106 µs        │ 100     │ 100
   ╰─ literal                             │               │               │               │         │
      ├─ arr8x8_batch1      9.373 µs      │ 24.64 µs      │ 11.07 µs      │ 11.29 µs      │ 100     │ 100
      ├─ arr8x8_batch64     45.37 µs      │ 85.98 µs      │ 64.06 µs      │ 63.24 µs      │ 100     │ 100
      ├─ arr8x8_batch256    171.2 µs      │ 264.9 µs      │ 220.1 µs      │ 219.9 µs      │ 100     │ 100
      ├─ arr64x64_batch1    4.091 ms      │ 15.55 ms      │ 7.339 ms      │ 7.437 ms      │ 100     │ 100
      ├─ arr64x64_batch64   8.533 ms      │ 23.33 ms      │ 12.4 ms       │ 12.7 ms       │ 100     │ 100
      ╰─ arr64x64_batch256  16.85 ms      │ 37.7 ms       │ 17.32 ms      │ 18.43 ms      │ 100     │ 100
```
