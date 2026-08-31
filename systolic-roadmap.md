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
    `FaultHook`. Slow, CPU. Used for validation and cross-checks, not for large
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

### `SyndromeModel` seam (per-array vs per-PE distributions)

There is a chance the sampled logic distribution must be built per PE (profile
each PE's logic-input distribution separately) rather than one distribution for
the whole array. Under the single-stuck-at fault model, only one PE is faulty
per run and `XorMaskHook` already targets one PE, so this is **not** a hook
change. It is only a question of *which* distribution you resolve before
building the fault.

Introduce a `SyndromeModel` with a single method
`distribution_for(pe) -> Distribution`:

- `PerArraySyndromeModel` ignores `pe` and returns the one global distribution.
- `PerPeSyndromeModel` indexes a table by PE coordinate.

The hook and the sampled lift consume an already-resolved distribution, so they
are shared entirely between the two. The only per-PE-specific work is the
model's storage/lookup and the offline generation mode (Phase 4). Design the
syndrome file format now with an **optional / broadcastable PE axis** so
per-array is per-PE with a single broadcast entry.

Both `SyndromeModel` kinds are built from the same per-PE profiling artifact
(Phase 2): per-array pools all PEs' observations, per-PE keeps them separate. So
the per-array-vs-per-PE choice is deferred to generation time and does not
change what profiling records.

Note: the only thing that would force per-PE *into* the hook is multiple
simultaneous faulty PEs, which the fault model excludes. If that ever changes,
`XorMaskHook` generalizes to a `target -> distribution` map.

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
dataset. This empirical distribution is what the syndrome generator (Phase 4)
runs the netlist over, so realistic syndromes cannot exist without it. It
depends on both the model (weights and how they map onto the array) and the
dataset (activations), so it is a per-(model, dataset, array) precompute and may
need regenerating per pair if the observed distributions differ significantly.
Placed here because it reuses Phase 1's infrastructure (SimulatedBackend, mapped
layers, dataset iteration) and reaches down to the array simulator. Independent
of Phase 3, so the two can proceed in parallel.

- **Rust.**
  - A recording hook implementing the existing `FaultHook<T>` trait
    (`crates/systolic/src/fault/hook.rs`). No new mechanism is needed:
    `multiply_add(index, activation, weight, partial_sum)` already receives
    exactly the PE coordinate and the three logic inputs. The hook records the
    triple keyed by `index`, then returns the correct passthrough
    `activation * weight + partial_sum`. "Disabled by default" just means not
    installing it (the array runs `NoFault`); profiling runs the
    `SimulatedBackend` with the recording hook installed.
  - Only the simulated array can profile. The lifted backend never materializes
    per-PE partial sums, so it has nothing to record. Profiling is therefore an
    oracle-side, SimulatedBackend-only precompute; it is slow, so subsample the
    dataset.
  - Data shape (open question, settle here): the key is always the PE
    coordinate, so the artifact is inherently per-PE. Two candidate value
    representations:
    - histogram `PE -> map[triple -> count]`: exact frequencies, but the joint
      triple space is enormous for f32 and needs quantization/binning. Viable
      only if the netlist consumes a quantized/fixed-point MAC representation.
    - reservoir sample `PE -> bounded list[triple]`: a bounded sample of raw
      observed triples; frequency is captured by sample density; composes with
      f32 natively; the generator replays the sample through the netlist.
      Recommended default for floats.
  - Because the key is the PE coordinate, the same artifact feeds both
    `SyndromeModel` kinds (per-array pools, per-PE keeps separate). The choice is
    deferred to generation time (Phase 4), not baked into profiling.
- **Bindings.** Expose the recording hook and a way to run the SimulatedBackend
  with it and extract the artifact (a "recording" mode on the backend that
  accumulates and can dump, or a dedicated `profile(...)` entry point).
- **Python.** A profiling driver that wraps the model with
  `BackendModel(model, SimulatedBackend(recording=True))`, runs it over a
  (subsampled) dataset, and serializes the per-PE artifact. Its identity /
  fingerprint is (model, dataset, array size), so it can be cached and matched to
  the generation step. CLI: a `systolic profile` command producing the artifact.
- **Validation.**
  - The recording hook is a passthrough: accuracy through the recording backend
    must equal the clean backend (recording changes nothing but observation).
  - Coverage: every PE used by the mapping appears in the artifact.
- **Done when.** `systolic profile` produces a cached, fingerprinted per-PE
  logic-input artifact for a (model, dataset, array) triple, ready for Phase 4's
  generator.
- **Watch out.**
  - Identity must include the array size, not just model+dataset: the mapping
    decides which activation/weight/output rows land on which PE, so the same
    model+dataset on a different array size produces different per-PE inputs.
  - Only the simulated array can profile; budget for its cost and subsample.
  - The float joint-triple cardinality drives the histogram-vs-sample choice.
    This is where the profiling format and the syndrome file format meet; settle
    both together.

### Phase 3: Logic faults, sampled path (workhorse)

**Objective.** Logic faults end to end on the fast sampled path, validated
against a synthetic distribution before the real generator exists.

- **Rust.**
  - Sampled simulation: `XorMaskHook` exists
    (`crates/systolic/src/fault/random.rs`); wire it into the backend dispatch.
    Note f32 does not implement `BitXor`; the mask application needs a bitcast
    to an integer of the same width.
  - Sampled lift: new, but reuses the accumulator lift structure. A syndrome
    is an XOR mask (`bits(correct_netlist_output) XOR bits(faulty_netlist_output)`
    for some observed input triple and fault case - see `XorMaskHook`, not an
    additive value). The lift still reduces to the accumulator case: at the
    faulty PE it already computes the correct value `V` the array would
    produce; apply the sampled mask to get `V' = bits(V) XOR mask`, take
    `delta = V' - V`, then propagate `delta` down the column exactly like
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
  - Syndrome file format with an optional/broadcastable PE axis.
- **Validation.** Agreement tooling: sampled-lifted vs sampled-simulated should
  match within float tolerance (both apply the same sampled masks; this is
  closer to exact than the netlist comparisons). Use a synthetic distribution.
- **Done when.** A logic (sampled) campaign runs end to end on both backends
  from the CLI using a synthetic distribution, and lifted vs simulated agree.
- **Watch out.** Sampling is stochastic (`XorMaskHook` draws from a weighted
  distribution). For agreement and reproducibility, seed the RNG and keep
  seeding out of the fault-selection Picker (two independent random sources:
  which fault, and which syndrome sample).

### Phase 4: Netlist simulator embedding

**Objective.** Fill the netlist path. Unlocks both the real syndrome generator
(feeds Phase 3) and the netlist simulated oracle.

- **Rust.**
  - Embed `crates/logic_simulation/` into `SimulatedMulAddHook::multiply_add`,
    which is currently `todo!()`
    (`crates/systolic/src/fault/simulated.rs`). The hook runs the PE netlist
    with a stuck gate for the given `case`.
  - Determine `sim_cases` for `SimulatedFaultContext` from the loaded netlist
    (the fault radix for netlist faults is `Index2::count * sim_cases`).
  - Syndrome generator: an offline routine that consumes the Phase 2 per-PE
    profiling artifact and runs the netlist over the observed logic inputs (per
    fault case) to build the sampled syndrome distribution. Per-array aggregation
    (pool all PEs' observations) first; per-PE (only if needed) is the same
    routine keyed by PE. Exports into the Phase 3 syndrome file format.
- **Python.** `LogicFault.netlist(target, case)`; netlist arm in
  `SimulatedBackend` only for now (the netlist *lift* is Phase 5). A path to
  export a generated distribution into the Phase 3 file format.
- **Validation.** Agreement tooling: sampled distribution (workhorse) vs
  netlist simulation (oracle). This is the measurement that tells us whether
  the global per-array distribution is accurate enough or whether per-PE is
  required. If per-PE is needed, implement `PerPeSyndromeModel` and the per-PE
  generation mode here; the runtime paths do not change (SyndromeModel seam).
- **Done when.** The netlist simulated oracle runs, real distributions can be
  generated and consumed by the Phase 3 sampled path, and we have a measured
  answer on per-array vs per-PE.
- **Watch out.** The netlist simulator may be much slower than the workhorse
  version; it is an oracle and a generator, not a campaign workhorse. Do not
  route large campaigns through it.

### Phase 5: Netlist lift

**Objective.** The remaining arm of the dispatch matrix: lifting a netlist
logic fault into matrix space.

- **Rust.** The netlist lift depends on the partial sum of the PEs *above* the
  faulty PE (unlike the sampled lift, which behaves like an accumulator fault).
  Work out the exact technique against the netlist oracle; the accumulator lift
  in `register_lift.rs` and its partial-sum computation (`for_activations`
  range, `slice` + `dot`) is the closest existing machinery.
- **Python.** Netlist arm in `LiftedBackend` and the torch applier.
- **Validation.** Agreement tooling vs the netlist simulated oracle from
  Phase 4.
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
| logic sampled               | `XorMaskHook`               | accumulator-like lift (Phase 3)    |
| logic netlist               | `SimulatedMulAddHook` (P4)  | partial-sum-dependent lift (P5)    |

## Open questions / deferred

- Profiling data shape: reservoir sample of raw triples (recommended for float)
  vs quantized histogram. Settle in Phase 2, jointly with the syndrome format
  since they share the same float-cardinality problem.
- Exact syndrome file format (PE axis representation, how masks + weights are
  stored). Settle at the start of Phase 3.
- Whether per-PE syndrome distributions are actually needed. Answered
  empirically in Phase 4 by the agreement measurement (sampled vs netlist
  oracle); do not build `PerPeSyndromeModel` before then, but keep the
  `SyndromeModel` seam and record per-PE data in Phase 2 so the option is free.
- The exact netlist lift technique (Phase 5). Deferred until the netlist oracle
  exists to validate against.
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
  (Phase 0, `_internal/systolic/layers.py`) restricts to `groups == 1` and
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
