# Systolic-array fault-injection experiment: roadmap

Working reference for the next few weeks of changes. Not permanent documentation.
Delete or fold into `docs/` once the work lands. Written for future Claude
sessions: each phase names the concrete files and types involved so a fresh
session can pick up without re-deriving the design.

## Goal

Add a new experiment to FaultForge that measures neural-network reliability
under permanent (stuck-at) faults in a weight-stationary systolic array. Two
fault families:

1. **Register faults** - stuck bits in a PE's weight, activation, or accumulator
   register. The Rust side (simulation and lifting) is essentially done; the
   work is the Python experiment, backends, and bindings.
2. **Logic faults** - faults in a PE's multiply-add logic. Not complete on the
   Rust side. Two sources: a sampled syndrome distribution (fast) and a full
   netlist simulation (exact). Added after register faults work end to end.

We run in **float32 (and maybe float16) only**. No integer path is planned.
Consequence: fault lifting is bit-exact only for integer arithmetic, so on
floats the lifted and simulated paths will differ in the low bits. We accept
this; the effect on inference accuracy is expected to be negligible. This is
why the agreement tooling (below) reports *similarity*, not equality, for the
float paths.

A third, cross-cutting component: **PE logic-input profiling**. Generating a
realistic syndrome distribution requires knowing the distribution of logic
inputs (activation, weight, incoming partial sum) each PE actually sees, which
depends on both the model (weights and how they map onto the array) and the
dataset (activations). Profiling is therefore a per-(model, dataset, array)
precompute that feeds syndrome generation. The data flow is: profile PE inputs
-> per-PE input distribution -> run the netlist over it -> syndrome distribution
-> sampled logic faults. It is introduced right after register faults because it
reuses that infrastructure and reaches down to the array simulator.

Sequence: get the full CLI -> Python -> Rust workflow running for register
faults first, then add logic faults as purely additive work.

## Reference material

- `docs/library.md` - the experiment framework this plugs into (`Experiment`,
  `ModelBundle`, `Fingerprint`, `Picker`, stop conditions, save/load).
- `docs/fault-lifting.md` - the theory behind register lifting. Read before
  touching any lift code. The three register cases (weight / activation /
  accumulator) and the "one recipe, every position" accumulator logic are the
  core.
- `experiments/encoded_memory/src/encoded_memory/experiment.py` - the one
  existing built-in experiment. Mirror its shape (results model, fingerprint,
  golden handling, serialize/deserialize, `_Display`).

## Architectural spine (decided)

These decisions are locked. Do not relitigate them without a reason.

### Fault is decoupled from Backend ("Option 1")

The fault taxonomy and the execution strategy are two independent axes.

- **Fault** = an execution-agnostic description of a single fault in array
  space. It knows its own enumeration (radix, and how to build itself from a
  fault id) but nothing about how it will be realized.
  - `RegisterFault` (target PE, register, bit, stuck value).
  - `LogicFault.sampled(target, ...)` - a sampled syndrome (later).
  - `LogicFault.netlist(target, case)` - a netlist case (later).
- **Backend** = the execution strategy only. Holds array geometry and a
  per-weight-shape `Mapping` cache. The ABC is `SystolicBackend`, not a generic
  "matmul backend" (see Conventions and terminology). Two implementations:
  - `SimulatedBackend` - the **oracle**. Runs the cycle-accurate array via a
    `PeHook`. Slow, CPU. Used for validation and cross-checks, not for large
    campaigns.
  - `LiftedBackend` - the **workhorse**. Asks Rust for a lifted *description*
    and applies it with torch ops (GPU, model dtype, amortized over the batch).
    This is what campaigns actually run on.

The two-axis dispatch (backend strategy x fault kind) is a set of match arms.
Adding logic later means: one new `Fault` type, one new arm in each backend,
and (for lifted) one new arm in the torch applier. The register path, the
backend classes, the experiment, and the CLI are never touched.

**Why not the alternatives.** A stateful backend design
(`make_faulty(id)` / `fault_radix()`) cannot express register-vs-logic from an
id alone, so it collapses into either this design or a per-combination grid. A
backend-per-(kind x execution) grid duplicates the mapping-cache and
conversion plumbing and multiplies with every new axis.

### Lifted = "Rust lifts, Torch applies"

Rust computes the lifted *description* (which weights / activation rows /
accumulator parts are affected); Python applies it in torch. This is the only
option that keeps the workhorse on GPU and in the model's float dtype and
amortized over the batch. It requires exposing the lifted structure across the
binding, not just a Rust matmul. `crates/systolic/src/fault/register_lift.rs`'s
`LiftedRegisterFault::matmul` is the reference for what each arm must compute;
it's already ported to torch in
`experiments/systolic/src/systolic/lift_apply.py`.

### One experiment, config selects

A single `SystolicFaultInjection` experiment. A config object picks the
mutually-exclusive fault kind (register / logic) and the sub-choices
(simulated / lifted; and for logic, sampled / netlist). This maximizes shared
golden / inference / scoring / serialization code. Register and logic are never
mixed in one instance.

Rough shape (names are suggestions):

```python
SystolicFaultInjection(
    bundle,                       # ModelBundle, reused as-is
    array=(nrows, ncols),
    fault=RegisterFaults(registers={Weight, Accumulator}),  # or LogicFaults(...)
    backend="lifted",             # or "simulated"
    metric=ReliabilityMetric.Accuracy,
)
```

### The oracle / workhorse ladder and agreement tooling

The oracle/workhorse split appears on two independent axes:

- **execution**: `SimulatedBackend` (oracle) vs `LiftedBackend` (workhorse).
- **logic source**: netlist simulation (oracle) vs sampled distribution
  (workhorse). The sampled distribution is *generated from* the netlist
  simulator offline; it is precompute that buys a large runtime speedup.

Because faults are decoupled from backends, the *same* fault set (same picker
seed / same fault ids) can be replayed through any backend and diffed. That is
the basis of the agreement tooling (see cross-cutting concerns). Equality is
only expected for integer arithmetic, which we do not use, so this tooling
reports similarity and time, not pass/fail. Strict equality stays where the
math guarantees it: the existing Rust proptests in
`crates/systolic/src/fault/register_lift.rs`.

## Cross-cutting concerns

These span multiple phases. Design the seams early even where the
implementation is deferred.

### Reuse from the existing codebase

- `Experiment` base (`faultforge.experiment`, backed by
  `src/faultforge/_internal/experiment.py`): `run` / `scores` / `serialize` /
  `deserialize` / `run_loop` / stop conditions. The new experiment subclasses
  this exactly like `EncodedFaultInjection`.
- `ModelBundle` (`faultforge.loading`): unchanged. Provides model + dataset +
  fingerprint.
- `Fingerprint` (`faultforge.fingerprint`): the new experiment builds one
  covering array size, fault config (including the register subset and syndrome
  model kind), backend, dtype, metric, and the bundle's fingerprint.
- `Picker` (`faultforge.Picker`): Fisher-Yates sampler over a dense fault
  radix, resumable via `Picker.from_returned`. This is how the experiment draws
  faults without replacement; identical pattern to
  `EncodedFaultInjection._inject_faults`.
- Layer mapping (`MappedLinear`, `MappedConv2d`, `BackendModel`): already
  implemented in `experiments/systolic/src/systolic/layers.py` and
  `model.py`. Keep the im2col decomposition in `MappedConv2d` and the
  transpose convention in `MappedLinear` (`backend.matmul(weight, x.T).T`). The
  mapped layers are the one boundary where PyTorch convention is bridged to the
  hardware convention; see Conventions and terminology.

### Package layout: standalone experiment package, not an `_internal` shim

Originally planned as `_internal/systolic/` plus a `faultforge/systolic.py`
shim - the convention `faultforge`'s own core modules still use internally
(see `src/faultforge/_internal/` and `AGENTS.md`). In practice systolic code
became its own uv workspace member, `experiments/systolic/` (package name
`systolic`), mirroring `experiments/encoded_memory/`: a flat `src/systolic/`
with its own `pyproject.toml`, its own Rust extension (`systolic._rust`,
bound to `crates/systolic_bindings/`), and its own CLI entry point
(`systolic = "systolic.main:main"`). It depends on `faultforge` as a normal
dependency rather than living inside it. New Phase 2+ modules go in
`experiments/systolic/src/systolic/` alongside `backend.py`, `fault.py`,
`experiment.py`, etc. - no `_internal` subpackage or shim needed there.

### Conventions and terminology: a systolic-array evaluator, not a generic matmul

The backend is not a generic matrix multiply; it is a weight-stationary
systolic-array evaluation. The distinction is load-bearing: the fault-lifting
recipes (`docs/fault-lifting.md`) are written in terms of the physical roles - a
stationary *weight* matrix, streamed *activations*, output rows tied to array
columns and activation rows to array rows. Renaming to `lhs`/`rhs` would erase
exactly the semantics the lift depends on. So:

- Keep the parameter names `weights` and `activations` (not `lhs`/`rhs`). They
  name the weight-stationary roles.
- Keep the hardware/array convention inside the backend and Rust:
  `weights (out_features, in_features)`, `activations (in_features, batch)`,
  result `(out_features, batch)`. This matches the SA simulator and the lift's
  row/column identities. Do not change the SA convention.
- Name the abstraction for what it is: `SystolicBackend` (the ABC), not "matmul
  backend". The method can stay `matmul` (it does compute a product) but its
  docstring must state the weight-stationary, column-activation convention.

PyTorch convention (row vectors, `y = x @ W.T`) is presented at exactly one
boundary: the mapped layers. `MappedLinear`/`MappedConv2d`
(`experiments/systolic/src/systolic/layers.py`) accept and return standard
PyTorch-shaped tensors and own the transpose to and from the backend
(`backend.matmul(weight, x.T).T`). So for anyone using `BackendModel` the
column/row difference is invisible, which is the goal. Do not spread the
transpose convention beyond the layer boundary.

Optional convenience: if direct callers (tests, the matmul-level agreement tool)
want a PyTorch-native entry point, add a thin `linear(weight, x)` helper
mirroring `torch.nn.functional.linear` (`x (batch, in)`, `weight (out, in)`,
returns `(batch, out)`) that transposes into the core `matmul`. Keep the core in
hardware convention.

### Register-subset restriction

Sometimes only the weight register is of interest, sometimes two of three,
usually all three. This is a **fault-space** concern and touches exactly one
layer: the config -> radix. Do the restriction as a dense re-index in Rust via
the `Space` context (`crates/systolic/src/space.rs`), not as Python-side reject
sampling.

- Add an allowed-register set to the fault-space context (extend `ArrayConfig`
  or wrap it) so `PeRegisterFault`'s `count` / `to_index` / `from_index`
  (`crates/systolic/src/fault/register.rs`) range over only the chosen
  registers. This keeps the radix honest and the `Picker`'s without-replacement
  exhaustion correct.
- The subset lives in the `RegisterFaults` config and is part of the
  fingerprint. A weight-only campaign must not resume into an all-registers one.
- Nothing else changes: not the hook (`RegisterHook`), the lift, the backend,
  or the mapped layers.

Default: all three registers.

### `SyndromeModel` seam (how finely the distribution is keyed)

There is a chance the sampled logic distribution must be built per PE rather
than as one distribution for the whole array. Under the single-stuck-at fault
model, only one PE is faulty per run and `XorMaskHook` already targets one PE,
so this is **not** a hook change. It is only a question of *which* distribution
you resolve before building the fault.

**Treat it as a keying function, not a binary.** `SyndromeModel` has one method,
`distribution_for(pe) -> Distribution`, and the granularities are points along
one axis rather than competing implementations:

| Keying | Groups (32x32 array) | Motivation |
|---|---|---|
| per-array | 1 | cheapest; the default |
| per-row | 32 | partial-sum magnitude grows with band depth |
| per-column | 32 | the natural keying for DRAIN |
| per-PE | 1024 | finest; only if the above prove insufficient |

Do **not** build these as separate classes. What actually differs is a group-by
key at generation time (`key(pe) -> ()` / `(y,)` / `(x,)` / `(y,x)`), whether
the row and column axes in the file are present or broadcast, and an array
index at lookup. The hook and the sampled lift consume an already-resolved
distribution and never learn which keying produced it. So design the syndrome
file format with **optional / broadcastable row and column axes**, and per-array
falls out as the degenerate case where both are broadcast - not a separate code
path. That is a handful of lines standing against a large cost difference,
which is the whole reason to keep the seam.

Row and column are not symmetric, and the asymmetry has structure worth
exploiting. Within ACTIVE, the activation at PE `(y,x)` is independent of `x` -
the same value shifts across the whole row - so column dependence enters only
through the weight, which varies by output channel and is broadly similar
across a trained layer. Row dependence is real: band depth sets how many
products the partial sum accumulates. DRAIN inverts this: all drain PEs in a column
see the identical partial sum for a given pass, so its natural keying is per-column.

**Why the default should be the coarsest keying that validates.** Generation
cost for one fault case is `K` netlist evaluations per distribution, so per-PE
costs `nrows x ncols` times more than per-array - 1024x on a 32x32 array. That
is not a storage concern, it is the fault-case budget: at `K = 1000` and a
budget of a few million evaluations, per-array characterizes thousands of fault
cases and per-PE characterizes a handful. Storage follows the same ratio.

Finer keying is also not automatically more accurate. At a *fixed* evaluation
budget `M` per fault case, per-PE estimates 1024 distributions from `M/1024`
samples each - a 32x larger standard error per distribution. Per-PE only wins
if the true between-PE difference exceeds roughly 32x the per-array sampling
noise. Below that bar it adds variance without removing bias. Per-row sits at a
much better point on that curve (`M/32` samples, ~5.7x the standard error)
while capturing the dominant structure, which is why it is the likely landing
spot rather than either endpoint.

Every keying is built from the same per-PE profiling artifact (Phase 2) by
reduction, so the choice is deferred to generation time and does not change
what profiling records.

Note: the only thing that would force per-PE *into* the hook is multiple
simultaneous faulty PEs, which the fault model excludes. If that ever changes,
`XorMaskHook` generalizes to a `target -> distribution` map.

**How the keying gets chosen, not guessed.** Because
Phase 2 profiling keeps each PE's sample separate rather than merging them
immediately, and keeps raw values rather than pre-aggregating, this is a real
statistics question, not intuition. It escalates through three tiers, each
cheaper and less conclusive than the next:

1. **Input-level, right after Phase 2, no netlist needed.** Compare a given
   PE's profiled `(activation, weight, partial_sum)` sample against the pooled
   sample over all PEs (per-marginal Kolmogorov-Smirnov, or a multivariate
   two-sample test - energy distance / MMD - on the joint triple). Necessary
   but not sufficient: if inputs are statistically indistinguishable per PE,
   syndromes must be too (same netlist, same inputs -> same outputs), so the
   coarsest keying is already exactly correct and a finer one buys nothing.
   If inputs do differ, that doesn't guarantee syndromes will - the netlist
   could still average the difference out - but it's a real, cheap early
   signal. Compare against per-row pooling as well as fully pooled, since
   per-row is the likely landing spot.
2. **Syndrome-level, once Phase 3 (syndrome generation) exists.** Because
   profiling retained raw per-PE samples, Phase 3's generator can produce a
   syndrome distribution at any keying from the identical underlying data
   (just choose the group-by key at replay time), and the same two-sample
   tests apply directly to syndromes instead of inputs. Doesn't need the
   array-embedded oracle (Phase 5) at all - just the generator. Stronger than
   tier 1, but still a distributional comparison, not a measurement of whether
   the difference actually matters for real campaigns.
3. **Model-level, once Phase 5 (netlist simulated oracle) exists.** Run
   matched campaigns at each candidate keying and compare all of them against
   the true netlist oracle via the agreement tooling. This is the real,
   practical answer: distributions can differ statistically (tier 2) without
   that difference changing accuracy predictions enough to matter. Weigh the
   result against the cost and bias-variance arithmetic above rather than
   simply picking the finest keying that differs.

Retaining per-PE keys and raw (non-aggregated) values in the Phase 2 artifact
is what makes all three tiers possible after the fact; pooling or aggregating
during profiling would throw away exactly the information they need.

### Phase 2/3 artifacts: content hash, not a fingerprinted recipe

`Fingerprint`/`raise_if_differs` exists to protect resumable, potentially
long-running work (`Experiment.run_loop`, `Picker.from_returned`) from
silently continuing under a drifted configuration. Phase 2 (profiling) and
Phase 3 (syndrome generation) are one-shot precomputes, not resumable
multi-step experiments - they don't have the failure mode `Fingerprint` was
built to catch, so neither gets its own `Fingerprint`. They're just scripts
that produce a file; caching, if wanted, is a plain deterministic
output-filename convention, not a `Fingerprint` diff.

Identity only actually matters where it always has: at the fault-injection
experiment (Phase 4), which *is* resumable. Its `Fingerprint` references the
syndrome artifact it loaded by a **content hash**, not by the parameters used
to generate it. This is deliberate and does more work than it looks like: a
content hash automatically captures the effect of the reservoir cap `K`, the
dataset-subsampling choice, the generation seed, or even a future
netlist-version change, without needing any of them tracked as separate
fingerprint fields - if any of them changes what the artifact actually
contains, the hash changes too. Nothing about *how* the artifact was produced
needs to be re-validated by the experiment consuming it, only *what* it
contains. The exact hashing mechanism (raw file bytes vs. a canonicalized
representation, to avoid false mismatches from incidental serialization
non-determinism) is deferred to whenever Phase 4 is actually implemented.

### Agreement and characterization tooling

Not unit tests (except where integer math guarantees equality). A harness that
replays a shared fault set through multiple backends and reports divergence and
time. Two granularities:

- **matmul-level**: compare `backend.matmul(w, a)` outputs for one fault on one
  weight shape. Cheap. Metrics: max abs / relative error, fraction of top-1
  argmax flips. Doubles as a tolerance-based regression test.
- **model-level**: run matched fault campaigns (same picker seed) per backend
  over the dataset, recording per-fault score, an output-divergence metric, and
  wall-time. Emit a table (reuse the CLI's dataframe/plot pattern in
  `faultforge_cli/encoded_memory/plots.py`).

First use in Phase 1 (lifted vs simulated register). Reused for every later
oracle/workhorse pair (sampled vs netlist, sampled-lifted vs sampled-simulated).

### Build reminder

After any Rust change, rebuild the affected extension(s) before running
Python:

```sh
.venv/bin/maturin develop -m pyproject.toml                          # faultforge._rust
.venv/bin/maturin develop -m experiments/systolic/pyproject.toml     # systolic._rust
```

Otherwise Python imports the stale compiled extension.

## Phases

### Phase 0: Python foundations (no faults yet)

**Objective.** Port the backend abstraction and layer mapping into FaultForge
conventions, running a clean (fault-free) matmul, so the model-wrapping
machinery is proven before faults enter.

**Status: done.**

- **Python.**
  - `experiments/systolic/src/systolic/backend.py`: `SystolicBackend` ABC (see
    Conventions and terminology; not a generic "matmul backend"). Single ABC,
    everything we need. **Minimal in Phase 0: just `matmul(weights,
    activations) -> Tensor` (hardware convention).** `set_fault(fault: Fault |
    None) -> None`, `nrows()`, `ncols()` need a `Fault` type and real array
    geometry, neither of which exist yet; they're added in Phase 1 alongside
    `SimulatedBackend`/`LiftedBackend`, the first backends that actually have
    both. `TorchBackend` (Phase 0's only implementor) has no array geometry or
    fault concept by design, so it would only ever fake-implement them.
  - A `TorchBackend` (always-correct, `weights @ activations`) for the golden
    path and as a trivial agreement baseline
    (`experiments/systolic/src/systolic/torch_backend.py`).
  - `experiments/systolic/src/systolic/layers.py`: `MappedLinear`,
    `MappedConv2d`. Keep im2col and the transpose convention.
    **`MappedConv2d` restricts to `groups == 1`, raising `ValueError`
    otherwise** - grouped/depthwise convolutions don't fit the current array
    model; see the "Grouped/depthwise convolutions" entry under Open
    questions / deferred.
  - `experiments/systolic/src/systolic/model.py`: `BackendModel` (recursive
    Linear/Conv2d replacement). Fault state lives on the backend, not the
    model. Mutates the wrapped model in place, same contract as
    `EncodedModule` - callers needing an untouched reference must
    `copy.deepcopy` before wrapping.
- **Validation.** `BackendModel(model, TorchBackend())` matches the unwrapped
  model's forward on a real bundle. This is the matmul-level agreement harness's
  first customer. Property tests in `experiments/systolic/tests/`;
  real-bundle accuracy check in `scripts/validate_torch_backend.py` (ResNet20 /
  CIFAR-10, since it's `groups=1` throughout - MobileNetV2/ShuffleNetV2
  variants of the `Cifar` bundle aren't usable until grouped convs are
  supported).
- **Done when.** A model wrapped with `TorchBackend` reproduces baseline
  accuracy through the FaultForge experiment scaffolding.
- **Watch out.** `MappedLinear` asserts 2D input and `MappedConv2d` asserts 4D;
  keep those. Bias handling differs between the two. `nn.Linear.bias`
  is `None` when `bias=False`.

### Phase 1: Register faults, end to end (the milestone)

**Objective.** Full CLI -> Python -> Rust workflow for register faults, on both
backends, with the register-subset restriction and the first agreement report.

- **Rust (mostly done; verify and extend).**
  - Register simulation: `RegisterHook` +
    `SystolicArray::with_hook` / `matmul` (done,
    `crates/systolic/src/fault/register.rs`, `array.rs`).
  - Register lifting: `Mapping::lift_register_fault` ->
    `LiftedRegisterFault` (done, `fault/register_lift.rs`). Study
    `LiftedRegisterFaultData` (`Weight` / `Activation` / `Accumulator` with
    `AccumulatorFaultPart`); the Python torch applier mirrors
    `LiftedRegisterFault::matmul` arm for arm.
  - Register-subset restriction: extend the `Space` context so the register set
    is configurable (see cross-cutting). New work.
- **Bindings (`crates/systolic_bindings/`, exposed to Python as the
  `systolic._rust` extension - kept separate from `faultforge._rust`).**
  - Expose `SystolicArray`, `Mapping` / `auto_mapping_for`, and a unified
    `Fault` py type (register variant to start) constructible from a fault id +
    fault-space context.
  - `SimulatedBackend` binding: `matmul(mapping, w, a, fault)` that matches the
    fault to a hook in Rust and returns the array result (numpy).
  - Lifted description: `mapping.lift(fault) -> LiftDescription`, a tagged
    Python object exposing the affected weights / rows / accumulator parts so
    the torch applier can consume them. Do **not** expose only the Rust matmul.
  - `fault_radix` over the (possibly restricted) fault space, for the `Picker`.
  - Update the `.pyi` stub at `experiments/systolic/src/systolic/_rust.pyi`.
- **Python.**
  - `SimulatedBackend` and `LiftedBackend` implementing the `Backend` ABC.
    `LiftedBackend.matmul` calls `mapping.lift(fault)` (cached per shape +
    fault) and applies the description in torch. See
    `experiments/systolic/src/systolic/lifted_backend.py` and `lift_apply.py`
    for the per-shape mapping/fault cache and the torch port of the Rust lift
    arms.
  - `experiments/systolic/src/systolic/fault.py`: the Python `Fault`
    wrapper(s) and the `RegisterFaults` fault-space config (holds the register
    subset, produces a `_rust.Fault` from an id, reports the radix).
  - `experiments/systolic/src/systolic/experiment.py`:
    `SystolicFaultInjection`. Mirror `EncodedFaultInjection`: golden handling,
    `ReliabilityMetric` reuse (`Accuracy` / `AccuracyDegradation` / `Sdc` /
    `Top1Sdc`), results pydantic model, `serialize`/`deserialize` with
    `Fingerprint.raise_if_differs`, `_Display`. Draw faults with `Picker` over
    the fault radix (use `Picker.from_returned` on resume, like
    `EncodedFaultInjection`).
- **CLI.** `experiments/systolic/src/systolic/commands.py` mirrors
  `encoded_memory`'s `commands.py`, with its own `results.py`/`plots.py`
  equivalents; registered through the `systolic` package's own typer app and
  entry point (`systolic = "systolic.main:main"`), not a shared CLI package.
  A `run` command (bundle, array size, register subset, backend, metric, stop
  conditions, save path) and a `compare` command that drives the agreement
  tooling (`systolic/agreement.py`).
- **Validation.**
  - Rust proptests already assert lifted == simulated for integers; keep green.
  - Agreement tooling: lifted vs simulated on a real model in float. Confirm
    divergence is negligible (the accepted float non-exactness) and record the
    time gap (expect the lifted workhorse to be far faster; see the benchmarks
    in `docs/fault-lifting.md`).
- **Done when.** `faultforge-cli systolic run ...` completes a register-fault
  campaign end to end on both backends, the register subset restricts the
  sampled faults correctly, and the agreement report shows lifted vs simulated
  agree within tolerance.
- **Watch out.**
  - A single physical fault lifts differently per layer (each weight shape has
    its own `Mapping`), so the effect must be recomputed per shape. Cache keyed
    by `(weight.shape, fault)`.
  - Float `RegisterHook` needs a `memory::BitBuffer` impl for f32 (f32 fault
    application already exists via `list_of_array_fault_f32`; confirm).
  - The array accumulates down columns in a fixed order, so simulated-float,
    lifted-float, and plain `w @ a` differ in low bits from float
    non-associativity. Expected, not a bug.

### Phase 2: PE logic-input profiling

**Objective.** Record, per PE, the distribution of logic-input triples
(activation, weight, incoming partial sum) that a given model sees over a given
dataset. This empirical distribution is what the syndrome generator (Phase 3)
runs the netlist over, so realistic syndromes cannot exist without it. It
depends on both the model (weights and how they map onto the array) and the
dataset (activations), so it is a per-(model, dataset, array) precompute and may
need regenerating per pair if the observed distributions differ significantly.
Placed here because it reuses Phase 1's infrastructure (mapped layers, dataset
iteration, and the `Mapping` that decides which PE sees what). Its output
feeds directly into Phase 3 (syndrome generation) as a hard dependency - that
phase cannot start without it. The sampled-path wiring (Phase 4) can still be
developed in parallel with both, since it only needs *a* distribution in the
agreed file format to build and unit-test against, not real profiled data.

- **Rust.**
  - A recording hook implementing the existing `PeHook<T>` trait
    (`crates/systolic/src/array/hook.rs`). No new mechanism is needed:
    `multiply_add(index, activation, weight, partial_sum)` already receives
    exactly the PE coordinate and the three logic inputs. "Disabled by
    default" just means not installing it (the array runs `NoOp`);
    profiling runs the `SimulatedBackend` with the recording hook installed.
  - **The hook must not record unconditionally.** `matmul` (`array.rs`) runs
    the array at its *full* physical `nrows x ncols` extent on every pass,
    zero-padding weight/activation outside the pass's active rectangle
    (`pass.range_y()/range_x()`). Most `multiply_add` calls are therefore not
    the PE doing real work, and recording them raw would contaminate the
    profile. Two independent facts control what to record.

    *Which cycle.* Deriving `run_shifted`'s output selection backwards through
    `unshift_output` (which picks `raw[r, x]` for `r` in
    `ncols-1-x .. ncols-1-x+batch_size`) and through the reversed `y` loop
    (which makes PE `(y,x)` feed PE `(y+1,x)` one cycle later) gives a single
    invariant:

    > PE `(y,x)` performs a genuinely-used multiply-add exactly on cycles `c`
    > where `0 <= c - x - y < batch_size`.

    Every other cycle is pipeline fill or drain. The same window falls out of
    the activation feed independently (the activation reaching PE `(y,x)` at
    cycle `c` is batch element `batch_size-1-(c-x-y)`), so this is a proof
    rather than a traced example.

    *Which regime.* Within a pass with `range_y = r0..r1` and
    `range_x = c0..c1`, a PE is in one of three input regimes. Note these are
    defined by band membership, so they hold for offset passes
    (`Pass::with_offset`) too, not just the corner-anchored ones
    `Mapping::auto` currently emits:

    | Regime | Where | MAC inputs | Reaches output? |
    |---|---|---|---|
    | FIRST | `y == r0`, `x in range_x` | `(real a, real w, 0)` | yes |
    | ACTIVE | `r0 < y < r1`, `x in range_x` | real `(a, w, partial_sum)` | yes |
    | ZERO | `y < r0`, or `x not in range_x` | `(0, 0, 0)` | above-band: yes; idle column: no |
    | DRAIN | `y >= r1`, `x in range_x` | `(0, 0, real partial_sum)` | yes |

    Keep "what are the inputs" and "does it reach the output" as **two
    separate axes**. Phase 2 only cares about the first; Phase 4 needs both.
    They genuinely diverge: an above-band PE has all-zero inputs yet is fully
    live, because fault-free it emits `0` into the band below it, and under a
    fault it emits garbage that propagates through the whole band into the
    result.

    DRAIN is the easy one to get wrong. Output is read from the *physical*
    bottom row (`array.rs:340`), not from row `r1`, so when a pass doesn't
    fill the array vertically the partial sums must travel down through the
    unloaded rows to reach readout. Those PEs compute `0.0 * 0.0 + partial_sum` on a
    fully-formed partial sum. Modelling them as an all-zero case would predict
    a fixed constant error where the real one is data-dependent. Their zero
    activation and weight are real, not an artifact - `matmul` explicitly
    zero-loads the array each pass (`pass_weights.fill`,
    `pass_activations.fill`).

    Conditioning on the regime rather than pooling over it is what makes this
    correct: at Phase 4 apply time the regime is a deterministic function of
    the mapping, not a random variable, so it is looked up structurally rather
    than sampled. That also removes the original worry that a
    frequently-idle PE would get a spuriously zero-heavy profile.

    FIRST is split out of ACTIVE for the same reason ZERO is: its `partial_sum` is
    exactly `0`, always, fault-free - everything above the band is zero-valued,
    and at `y == 0` the code takes the `T::zero()` branch outright. That is a
    degenerate point, not a narrow distribution, and it is deterministic from
    the mapping like every other regime. Pulling it out means the only
    remaining row-dependence inside ACTIVE is the smooth growth of `partial_sum`
    magnitude with band depth (a sum of `y - r0` products, so roughly `sqrt(d)`
    in scale), which is far more poolable than a mixture of "exactly zero" and
    "not zero" would be.

    Whether idle columns *should* contribute to the accumulated result is a
    separate, deliberately deferred question about array semantics. Today
    `matmul` slices `pass_result` by `range_x` before the `add_assign`, so
    they don't; real hardware might accumulate all columns uniformly to avoid
    mapping-aware control, and it only matters in the presence of faults.
    Nothing here depends on the answer: ZERO stores no profiling data, so
    flipping that decision later changes only Phase 4's "reaches output" axis,
    and the Phase 2 artifact is unaffected.
  - **Cycle tracking needs no trait change.** Every PE gets exactly one
    `multiply_add` per cycle, so a per-PE call counter *is* the cycle index.
    Reset it per `run()` and the filter is `0 <= n - x - y < batch_size`
    against that counter. The hook additionally needs the pass rectangle and
    `batch_size`, which the profiling driver supplies per pass - so the driver
    owns the pass loop rather than `matmul` notifying the hook, and existing
    array code stays untouched.
  - **One implementation, not an oracle/workhorse pair.** These triples are
    also derivable analytically without cycling the array (per pass, an
    exclusive `cumsum` of `W_pass[:, :, None] * A_pass[None, :, :]` along the
    reduction axis gives every partial sum at once), and that would be perhaps 10x
    faster. It is deliberately **not** being built now:
    - Profiling is a one-time precompute at a scale the array handles fine
      (see the scale note under Python), so the speedup buys nothing today.
      This is nothing like the ~1000x that justifies the
      `SimulatedBackend`/`LiftedBackend` split.
    - Phase 5's netlist oracle embeds into `SimulatedMulAddHook::multiply_add`,
      which receives *exactly* the same `(index, activation, weight,
      partial_sum)` this hook receives. Profiling through the hook makes the
      profiled distribution identical to what the netlist will later be fed -
      by construction, through the same call site, not by derivation.
    - The two would encode the same derivation of the dataflow, so checking
      them against each other catches transcription errors but not the failure
      that matters (a misunderstanding of the dataflow). Real validation comes
      from the checks below instead.
    - Ordering: the hook is what would *validate* a future tensor path. Built
      first, it keeps that option cheap; built second, the tensor path would
      have shipped with nothing behind it.

    If profiling cost ever becomes a real problem, add the tensor path then,
    with this hook as its oracle. That is the genuine oracle/workhorse
    relationship, in the right order.
  - Subsample the dataset. Selection should be **uniform random, not
    stratified**: the Goal is a *realistic* syndrome distribution, meaning it
    should mirror the real
    frequency the deployed model sees things at, not an artificially
    rebalanced view - stratifying by class would work against that. Subsample
    size is a run parameter like `K`, not a hardcoded constant, and its seed
    is a local reproducibility knob only (see "content hash, not a
    fingerprinted recipe").
  - Data shape (settled): `PE -> bounded reservoir sample of raw triples`, not
    a histogram. A histogram needs binning to collapse repeats into counts,
    which only pays off if the input space is small/discrete enough for real
    collisions to occur; nothing in the codebase quantizes MAC inputs (no
    `FixedPoint` type anywhere, and the one existing netlist-wiring precedent,
    `crates/float_mac_faults`, feeds raw `f32::to_bits()` straight into the
    netlist), and the project is float32-only with no integer path planned
    (see Goal). So the joint triple space is the full float32^3 space, and a
    histogram over it degenerates into one bucket per observation - a strictly
    worse reservoir sample. A bounded uniform reservoir per PE needs no
    binning, composes with f32 natively, and captures frequency through sample
    density: commonly-occurring triples are proportionally more likely to be
    retained, so replaying the sample through the netlist (Phase 3) reproduces
    a realistic syndrome distribution without evaluating the netlist on every
    single observation.
  - Because the key is the PE coordinate, the same artifact feeds both
    `SyndromeModel` kinds (per-array pools, per-PE keeps separate). The choice is
    deferred to generation time (Phase 3), not baked into profiling.
- **Bindings.** Expose the recording hook and a way to run the SimulatedBackend
  with it and extract the artifact (a "recording" mode on the backend that
  accumulates and can dump, or a dedicated `profile(...)` entry point).
  Anything that needs to reason about regimes from Python (the Phase 4 lift,
  the keying-divergence analysis, a future tensor path) also needs the
  mapping's passes readable there: `Pass` is already a `#[pyclass]` with all
  four fields exposed as getters (`crates/systolic_bindings/src/mapping.rs`),
  but `Mapping` has no `passes` accessor yet. Small addition.
- **Python.** A profiling driver that runs the model over a (subsampled)
  dataset through the recording hook and serializes the per-PE artifact. CLI: a
  `systolic profile` command producing the artifact.

  **Scale.** `MappedConv2d` im2cols and tiles patches, so a conv layer's matmul
  free axis is `batch_size x output_positions`, not `batch_size`. ResNet20 is
  ~40M MACs per image, and the array's overhead over that is modest - a
  representative stage-1 conv (patch 144, 16 out channels, 32x32 array) is 5
  passes x 1024 PEs x 1086 cycles = 5.56M PE-cycles against 2.36M useful MACs,
  so roughly **2.4x**, since `run_shifted`'s pipeline fill amortizes over the
  free axis. That puts profiling at a few seconds per image, so **a few dozen
  images run in minutes** - fine for a one-time precompute, and the reason the
  tensor path isn't needed.

  It also means the dataset subsample can be *very* small: a few dozen images
  already saturate the reservoirs even at `K` in the tens of thousands, so
  subsample size is really a knob for dataset diversity, not for sample count.

  Profiling is a one-shot precompute,
  not a resumable multi-step experiment, so it does **not** need its own
  `Fingerprint`/`raise_if_differs` - that machinery exists specifically to
  protect `Experiment.run_loop`/`Picker`-style resumable work from silently
  continuing under a drifted config, which isn't a failure mode a one-shot
  script has. Caching, if wanted, can be a plain deterministic output-filename
  convention instead. The reservoir cap per PE (`K`) is a run parameter
  (CLI-exposed, no hardcoded default baked into the code), not a fixed
  constant - see the `SyndromeModel` seam and the "content hash, not recipe"
  note below for why it doesn't need to be tracked anywhere beyond that.
  Storage format (settled): since `K` is one run parameter, not per-PE
  adaptive, the whole artifact is a dense array, not a dict of variable-length
  per-PE lists. One array per regime that needs data:

  - `first_triples: (nrows, ncols, K, 3)` + `first_fill: (nrows, ncols)`.
    Kept 3-wide rather than 2-wide so it shares every code path with ACTIVE;
    the partial-sum column is structurally zero and compresses away.
  - `active_triples: (nrows, ncols, K, 3)` + `active_fill: (nrows, ncols)`
  - `drain_partial_sums: (nrows, ncols, K)` + `drain_fill: (nrows, ncols)`
  - ZERO stores nothing. It is a single deterministic input point `(0, 0, 0)`,
    so Phase 3 evaluates the netlist there once per fault and gets one
    syndrome.

  The `*_fill` arrays are load-bearing, not diagnostics: they hold
  `min(K, n)`, without which an unfilled reservoir slot is indistinguishable
  from a real sample and would be sampled as zero-padding. Storing the *total*
  observation count `n` per PE as well is free (Algorithm R tracks it anyway)
  and useful for spotting a PE with empty support, but nothing depends on it -
  Phase 4 does not need it to weight anything, since regime membership is
  looked up structurally from the mapping.

  Save via plain `numpy.savez_compressed` (`experiments/systolic` already
  depends on `numpy`; no need for HDF5/Parquet) with a small JSON sidecar for
  human-facing metadata (model, dataset, array size, `K`, subsample
  size/seed) - informational only, not used for fingerprinting or caching.
  `np.savez_compressed` is worth it by default: weight-stationary reuse means
  a lot of triples share the same weight value, so there's real redundancy,
  and profiling is dominated by compute, not I/O.
- **Validation.** The hook takes its *values* straight from the simulator, so
  those are correct by construction. The **filter** is not - it is the derived
  reasoning above, and it is what these checks exist to pin down. Its failure
  mode is unusually benign, which is what makes them cheap: outside the cycle
  window the incoming activation is zero (either `zero_activations` or
  `shift_activations`' padding) and the partial sum from above carries the same
  wavefront index, so it is zero too. **A filter bug cannot record wrong
  values, only spurious `(0,0,0)` ones.**
  - Passthrough: accuracy through the recording backend must equal the clean
    backend (recording changes nothing but observation).
  - Count: each ACTIVE PE records exactly `batch_size` observations per pass.
    Catches cycle-window *width* errors immediately, and is checkable by direct
    enumeration over `(y, x, cycle)` without running the simulator at all.
  - All-zero census: `(0,0,0)` triples should be essentially absent from ACTIVE
    reservoirs. Post-ReLU `a == 0` is common, but `a == 0 && w == 0 &&
    partial_sum == 0` together is not in a trained net. Catches window *offset*
    errors, which the count check alone would miss.
  - Reconstruction: for each pass, column `x`, free-axis element `b`, the last
    band row's recorded `partial_sum + a*w` must equal `pass_output[output_row(x), b]`
    from an independent matmul. Validates the partial-sum chain endpoint, the regime
    classification, and the column-to-output-row mapping against a computation
    that shares no reasoning with the profiler.
  - Coverage: every PE used by the mapping appears in the artifact, and each
    PE's FIRST/ACTIVE/DRAIN fill counts match the regimes the mapping puts it
    in. Include offset passes (`Pass::with_offset`), not just the
    corner-anchored ones `Mapping::auto` emits today.
  - Keying divergence (cheap, no netlist needed): compare each PE's input
    sample against the pooled sample over all PEs, and against a per-row
    pooling (e.g. a Kolmogorov-Smirnov test per marginal, or a multivariate
    two-sample test - energy distance / MMD - on the joint triple). This is a
    necessary but not sufficient signal for how finely the syndrome model must
    be keyed: if input distributions are statistically indistinguishable
    across PEs, syndrome distributions must match too (same netlist, same
    inputs), so the coarsest keying is already exactly right. If inputs do
    differ, syndromes might still end up similar (the netlist can average
    differences out) - the real answer only comes from the syndrome-level
    comparison in Phase 3. See the `SyndromeModel` seam section for the full
    three-tier methodology. **This is the one part of Phase 2 that is not on
    the critical path**: Phase 3 needs only the artifact, and the keying
    choice isn't made until Phase 3 generation time anyway.

**Suggested chunking.** This phase is roughly 600-700 lines of implementation
plus tests, across two languages and an FFI boundary - more than one review
pass, and the risk is concentrated in a small part of it (the filter). Each
chunk below has a stable interface and a real test, so none of them needs
throwaway scaffolding to be reviewable on its own.

1. **Regime + cycle classification.** The pure function `(pass, array size,
   batch_size, y, x, cycle) -> Regime | Skip`. No reservoir, no hook, no
   simulator. *Done when* hand-computed small cases, offset-pass cases and the
   count invariant pass - the last by direct enumeration over `(y, x, cycle)`,
   which needs no simulation.
2. **Per-PE reservoir.** Algorithm R, seeded RNG, fill and total counts, one
   per regime. *Done when* uniformity over a known stream, determinism under a
   fixed seed, and the `n < K` case are tested. Independent of (1) - either
   order.
3. **Recording hook + per-pass driver.** Composes (1) and (2) into the
   `PeHook<f32>` with the per-PE cycle counter, plus the driver owning the
   pass loop. *Done when* one matmul produces an in-memory artifact with the
   passthrough, all-zero census and reconstruction checks passing.
4. **Bindings + serialization.** Expose to Python; `npz` plus JSON sidecar.
   *Done when* a round-trip test passes and the artifact layout is fixed.
5. **Model-level driver + `systolic profile`.** Walk the mapped layers,
   accumulate across layers and batches, CLI. *Done when* the phase's "Done
   when" below is met.
6. **Keying-divergence analysis.** Parallel track, not a gate - Phase 3 can
   start once (5) lands.

Chunks 1-3 carry the conceptual risk and deserve careful review; 4-5 are
plumbing. Explicitly out of scope: the tensor path (deferred optimization, with
chunk 3 as its future validation target), the syndrome-model keying choice
(a Phase 3 generation-time argument), and the idle-column accumulation question
(see Open questions).

- **Done when.** `systolic profile` produces a per-PE logic-input artifact for
  a (model, dataset, array) triple, ready for Phase 3's generator. Not
  "fingerprinted" - see the one-shot-precompute note above.
- **Watch out.**
  - Identity must include the array size, not just model+dataset: the mapping
    decides which activation/weight/output rows land on which PE, so the same
    model+dataset on a different array size produces different per-PE inputs.
  - Don't filter observations on *values*. "Skip when activation and weight are
    both zero" looks equivalent to the regime rule and is wrong in both
    directions: post-ReLU activations are exactly `0.0` very often and those
    are genuine ACTIVE observations, and it would also swallow all of DRAIN.
    The filter has to be structural (pass, index, cycle).
  - The reservoir cap directly bounds Phase 3's netlist-evaluation cost
    (`O(PEs x cap x fault_cases)`), so it isn't just a Phase 2 storage
    decision - pick it with Phase 3's cost in mind.

### Phase 3: Syndrome generation from sampled inputs

**Objective.** Turn Phase 2's per-PE profiled inputs into the syndrome
distribution Phase 4's sampled fault path consumes, by running the netlist
over them. This needs only the netlist crate and Phase 2's artifact - no
`SystolicArray`, no `PeHook`, no array structure involved at all - so it's
buildable independent of the array-embedded oracle (Phase 5). A real PE
netlist is available, so there's no need for a synthetic placeholder
distribution anywhere downstream of this phase.

- **Rust.**
  - The standalone netlist-evaluation function, built on
    `crates/logic_simulation`: takes a fault case and an `(activation, weight,
    partial_sum)` triple, loads the PE netlist, and returns the corrupted
    output. This is the one real prerequisite for this phase, and the only
    thing that touches the netlist directly.
  - A syndrome is an XOR mask:
    `bits(correct_netlist_output) XOR bits(faulty_netlist_output)`, computed
    for a specific observed input triple and fault case (see `XorMaskHook`,
    `crates/systolic/src/fault/random.rs` - not an additive value; Phase 4's
    lift derives an additive delta from this mask at apply time, but the
    syndrome itself is the mask).
  - Syndrome generator: an offline routine that consumes the Phase 2 per-PE
    profiling artifact and calls the evaluation function above over the
    observed logic inputs (per fault case) to build the sampled syndrome
    distribution (mask -> occurrence weight). Per-array aggregation (pool all
    PEs' observations) first; per-PE (only if needed, see `SyndromeModel`
    seam) is the same routine keyed by PE. Exports into the syndrome file
    format Phase 4 consumes.
  - **One distribution per input regime**, mirroring Phase 2's artifact: an
    ACTIVE distribution from the sampled triples, a DRAIN distribution from the
    sampled `(0, 0, partial_sum)` inputs, and for ZERO a single syndrome per fault
    case from the one deterministic `(0, 0, 0)` input - no sampling, no
    reservoir. Keeping them separate is what lets Phase 4 condition on the
    regime instead of marginalizing over it.
- **Python.** A path to export a generated distribution into the syndrome
  file format; a `systolic generate` (or similar) CLI command that runs
  generation from a cached Phase 2 artifact.
- **Validation.** The syndrome-level tier of the `SyndromeModel` seam's
  three-tier methodology: because the generator takes the keying as a group-by
  argument, distributions at several keyings can be generated from the same
  profiled inputs and compared with a two-sample test (energy distance / MMD,
  or per-mask frequency comparison), giving a stronger-than-input-level signal
  on how finely the model needs to be keyed. This doesn't need the
  array-embedded oracle (Phase 5) - the real, practical answer (does the choice
  affect campaign accuracy) still waits for Phase 5's model-level agreement
  tooling.
- **Done when.** Given a Phase 2 profiling artifact and a real PE netlist, a
  `systolic` command produces a syndrome file that Phase 4 can load and
  sample from.
- **Watch out.** The netlist simulator is expected to be much slower than the
  array simulator; this is exactly why the Phase 2 reservoir cap matters -
  generation cost is `O(PEs x reservoir_cap x fault_cases)`, not
  `O(PEs x dataset_size x fault_cases)`.

### Phase 4: Logic faults, sampled path (workhorse)

**Objective.** Logic faults end to end on the fast sampled path, using the
real syndrome distribution Phase 3 generates.

- **Rust.**
  - Sampled simulation: `XorMaskHook` exists
    (`crates/systolic/src/fault/random.rs`); wire it into the backend dispatch.
    Note f32 does not implement `BitXor`; the mask application needs a bitcast
    to an integer of the same width.
  - **Regime is looked up, not sampled.** For a given pass, which of
    ACTIVE/ZERO/DRAIN a PE is in is a deterministic function of the mapping
    (`pass.range_y()`/`range_x()`; see Phase 2's regime table), so the hook and
    the lift select the matching Phase 3 distribution structurally, then sample
    within it. This is also where Phase 2's second axis lands: whether a
    corrupted value *reaches the output* is separate from what the PE's inputs
    were. An above-band ZERO-regime PE is fully live (its garbage propagates
    down through the band), while an idle-column PE currently is not, because
    `matmul` slices `pass_result` by `range_x` before accumulating. If that
    slicing is ever changed to accumulate all columns uniformly, only this
    lookup changes - no reprofiling, no regeneration.
  - Sampled lift: new, but reuses the accumulator lift structure. The syndrome
    is an XOR mask (see Phase 3 for its definition), not an additive value.
    The lift still reduces to the accumulator case: at the faulty PE it
    already computes the correct value `V` the array would produce; apply the
    sampled mask to get `V' = bits(V) XOR mask`, take `delta = V' - V`, then
    propagate `delta` down the column exactly like
    `LiftedRegisterFaultData::Accumulator` does (one output row, propagate down
    the column) - the column-propagation machinery only needs a delta, it
    doesn't care that this one is derived from a bit-mask rather than a
    register stuck-at. Reuse `AccumulatorFaultPart`'s column-propagation
    machinery, swapping "corrupt the partial sum via stuck-at" for "corrupt it
    via the sampled XOR mask, then take the delta". See
    `lift_accumulator_fault` in `register_lift.rs`. An exponent-bit flip can
    legitimately produce `Inf`/`NaN`, but this needs no special handling here:
    `agreement.py`'s `_compare` already treats matching `Inf`/`NaN` on both
    backends as agreement (inherited from register faults, which can saturate
    the same way).
- **Python.**
  - `LogicFault.sampled(target, distribution)` and the `LogicFaults`
    fault-space config, which holds a `SyndromeModel` (per-array impl first;
    the seam supports per-PE later). Producing a fault resolves
    `model.distribution_for(target)` and hands the concrete masks/weights to the
    `_rust.Fault`, so the hook and lift only ever see a resolved distribution.
  - New arms in `SimulatedBackend` / `LiftedBackend` dispatch and in the torch
    applier. Register code untouched.
  - Load the syndrome file format Phase 3 produces (optional/broadcastable PE
    axis, per the `SyndromeModel` seam).
  - `SystolicFaultInjection`'s `Fingerprint` includes a content hash of the
    loaded syndrome artifact (see "Phase 2/3 artifacts: content hash, not a
    fingerprinted recipe" in Cross-cutting concerns) - this is the one place
    in the whole logic-fault pipeline where identity is actually tracked.
- **Validation.** Agreement tooling: sampled-lifted vs sampled-simulated should
  match within float tolerance (both apply the same sampled masks; this is
  closer to exact than the netlist comparisons).
- **Done when.** A logic (sampled) campaign runs end to end on both backends
  from the CLI using a real generated distribution, and lifted vs simulated
  agree.
- **Watch out.** Sampling is stochastic (`XorMaskHook` draws from a weighted
  distribution). For agreement and reproducibility, seed the RNG and keep
  seeding out of the fault-selection Picker (two independent random sources:
  which fault, and which syndrome sample).

### Phase 5: Netlist simulated oracle

**Objective.** Embed the netlist into the array simulator itself, so
`SimulatedBackend` can run a real model campaign with a genuinely
gate-accurate fault at one PE - the "logic netlist" oracle in the dispatch
matrix, and the ground truth for model-level agreement validation against the
sampled/lifted approximations. Reuses the same netlist-evaluation function
Phase 3 built, just wired into a different place; **not** a prerequisite for
Phase 3 or Phase 4, so it can be built in parallel with, or after, either.

- **Rust.**
  - Embed the Phase 3 netlist-evaluation function into
    `SimulatedMulAddHook::multiply_add`, currently `todo!()`
    (`crates/systolic/src/fault/simulated.rs`), so the hook runs the PE
    netlist with a stuck gate for the given `case` inside a real cycle-accurate
    array run.
  - Determine `sim_cases` for `SimulatedFaultContext` from the loaded netlist
    (the fault radix for netlist faults is `Index2::count * sim_cases`).
- **Python.** `LogicFault.netlist(target, case)`; netlist arm in
  `SimulatedBackend` only for now (the netlist *lift* is Phase 6).
- **Validation.** Agreement tooling: sampled distribution (workhorse) vs
  netlist simulation (oracle), at the model level - the model-level tier of
  the `SyndromeModel` seam's three-tier methodology, and the ground-truth
  check that the sampled path (Phases 3-4) is accurate enough. If a finer
  keying is needed, it is a group-by argument to Phase 3's generator plus the
  matching axes in the syndrome file; the runtime paths do not change
  (SyndromeModel seam).
- **Done when.** The netlist simulated oracle runs a real model campaign, and
  agreement against the sampled path is measured, giving a final answer on how
  finely the syndrome model needs to be keyed.
- **Watch out.** The netlist simulator may be much slower than the workhorse
  version; it is an oracle, not a campaign workhorse. Do not route large
  campaigns through it.

### Phase 6: Netlist lift

**Objective.** The remaining arm of the dispatch matrix: lifting a netlist
logic fault into matrix space.

- **Rust.** The netlist lift depends on the partial sum of the PEs *above* the
  faulty PE (unlike the sampled lift, which behaves like an accumulator fault).
  Work out the exact technique against the netlist oracle; the accumulator lift
  in `register_lift.rs` and its partial-sum computation (`for_activations`
  range, `slice` + `dot`) is the closest existing machinery.
- **Python.** Netlist arm in `LiftedBackend` and the torch applier.
- **Validation.** Agreement tooling vs the netlist simulated oracle from
  Phase 5.
- **Done when.** All four logic cells of the dispatch matrix
  (sampled/netlist x simulated/lifted) exist and agree with their oracles
  within tolerance.

## Dispatch matrix (target end state)

Fault kind is the row (a `Fault` type); backend is the column. Each cell is one
match arm. This is the whole surface that grows as features land.

| fault                       | SimulatedBackend (oracle)   | LiftedBackend (workhorse)          |
|-----------------------------|-----------------------------|------------------------------------|
| register weight             | `RegisterHook`              | weight lift (done in Rust)         |
| register activation         | `RegisterHook`              | activation lift (done in Rust)     |
| register accumulator        | `RegisterHook`              | accumulator lift (done in Rust)    |
| logic sampled               | `XorMaskHook`               | accumulator-like lift (Phase 4)    |
| logic netlist               | `SimulatedMulAddHook` (P5)  | partial-sum-dependent lift (P6)    |

## Open questions / deferred

- Exact syndrome file format (broadcastable row/column axis representation, how
  masks + weights are stored). Settle at the start of Phase 3.
- How finely the syndrome model needs to be keyed (per-array / per-row /
  per-column / per-PE). See the `SyndromeModel` seam section for the full
  three-tier methodology: a cheap early signal from Phase 2 (input
  distributions, no netlist needed), a stronger signal from Phase 3 (syndrome
  distributions, no array oracle needed), and the real practical answer from
  Phase 5's model-level agreement measurement (sampled vs netlist oracle).
  Default to the coarsest keying until evidence says otherwise - finer keying
  costs a factor of the group count in generation *and* raises per-group
  variance at fixed budget. Keep the seam and record per-PE data in Phase 2 so
  every option stays free.
- The exact netlist lift technique (Phase 6). Deferred until the netlist oracle
  exists to validate against.
- **Should idle columns contribute to the accumulated result?** Today `matmul`
  slices `pass_result` by `pass.range_x()` before the `add_assign`, so a column
  outside the pass's band contributes nothing. Real hardware might instead
  accumulate all `ncols` columns uniformly, since that needs no mapping-aware
  control. It is an arbitrary modelling choice that is only observable in the
  presence of faults, and under `Mapping::auto` it is not observable at all (a
  pass's idle columns map past the end of its `output_rows`, so they'd have
  nowhere to land). It becomes observable under hand-built mappings, which is
  exactly what `Pass::with_offset` is being kept for. Keep the current
  behaviour until there is a reason to change it. Deliberately does not block
  Phase 2: the ZERO regime stores no profiling data, so flipping this later
  changes only Phase 4's regime lookup.
- Multiple simultaneous faulty PEs. Out of scope. Noted only because it is the
  one change that would push per-PE distributions into the hook itself.
- **Grouped/depthwise convolutions (`groups != 1`) don't fit the current
  array model.** A grouped conv is block-diagonal: `G` independent
  sub-matmuls (`C_in/G` -> `C_out/G` each), not one dense matmul over all
  channels - this is a fact about the math, not an implementation gap.
  `Mapping`/`Pass` (`crates/systolic/src/array/mapping.rs`) only represents
  one dense, fully-connected block per pass (`Mapping::validate` requires
  every activation row in a pass to connect to every output row in that
  pass), so there's currently no way to place several independent sub-blocks
  in the same array/pass without materializing the full block-diagonal
  weight matrix - which wastes `G-1` out of every `G` PE cells (nearly the
  whole array for depthwise, where `G = C_in = C_out`). `MappedConv2d`
  (Phase 0, `experiments/systolic/src/systolic/layers.py`) restricts to
  `groups == 1` and
  raises `ValueError` otherwise; real grouped-conv models in the `Cifar`
  bundle (MobileNetV2, ShuffleNetV2) are unsupported until this is resolved.
  Options considered when this was found:
  1. Loop one `backend.matmul` call per group (correct, but `G` sequential
     calls per layer - for depthwise that's hundreds of tiny calls per
     layer, and each call gets more expensive once it crosses into
     `SimulatedBackend`'s cycle simulation or `LiftedBackend`'s Rust calls
     in Phase 1, so this compounds badly at campaign scale).
  2. Materialize the block-diagonal weight matrix and run one dense matmul
     (correct, wastes most of the array on multiply-by-zero for small
     groups/depthwise).
  3. Extend `Mapping`/`Pass` to represent independent sub-blocks sharing
     array space (the "real" fix, but a genuine array-model design
     question - bigger than a layer-mapping detail, needs its own
     discussion).
  Needs a dedicated design discussion before any grouped-conv model can be
  validated end to end.
