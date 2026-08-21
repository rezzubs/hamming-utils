# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Experiments live in the repository under `experiments/` as their own packages,
each pinned to a specific `faultforge` version and installed from a pinned git
ref rather than published to PyPI. This is meant to keep the library itself
small and its dependencies minimal regardless of how many experiments accumulate
over time, while letting an individual experiment stop being actively maintained
(e.g. once its paper is published) without constraining the design of the
library.

### Changed

- **Repository layout**: `faultforge`'s Python source now lives directly
  at the repository root (`src/`, `tests/`) rather than nested under
  `packages/faultforge`/`faultforge/` - installing from source no longer needs
  a `#subdirectory=faultforge` suffix. `encoded_memory` (previously bundled
  into `faultforge` as `faultforge.experiments.encoded_memory`) is now its own
  package, `experiments/encoded_memory`, pinned to an exact `faultforge` version
  rather than tracking it. (#30)
- **`encoded-memory` CLI**: what was `faultforge-cli`'s `faultforge
  encoded-memory <command>` is now `encoded-memory <command>`, shipped as
  part of the `experiments/encoded_memory` package instead of a separate
  `faultforge-cli` package. (#30)
- **File/loading/dataset helpers moved out of the root namespace**:
  `DEFAULT_BATCH_SIZE`, `DEFAULT_DEVICE`, `DeviceLike` move to
  `faultforge.dataset`; `DEFAULT_DTYPE` moves to `faultforge.loading`;
  `AnyPath`, `is_compressed`, `open_text` move to a new `faultforge.io`. These
  aren't part of the framework's primary surface (`Experiment`, `Fingerprint`,
  `ModelBundle`, ...) - just convenience helpers, so they belong next to the
  module that actually explains them rather than crowding the root namespace.
  (#30)

### Removed

- **`faultforge-cli`** is no longer published to PyPI or maintained as a
  separate package - see `encoded-memory` above. (#30)

## [0.2.1] - 2026-07-08

`0.2.0`'s release pipeline failed to publish `faultforge` and never
completed, so it's a dead/uninstallable version - `faultforge-cli==0.2.0`'s
`faultforge` dependency was also left unconstrained instead of pinned to a
matching version. `0.2.1` is the first fully published, correctly paired
release.

## [0.2.0] - 2026-07-08

0.2 is a more or less full rewrite of both the Rust and Python layers. If you
used 0.1's `System`-based API (`faultforge.system.System`,
`faultforge.cifar.system`, `faultforge.imagenet.system`, `EncodedSystem`), none
of that exists anymore. The Python package now lives in a `packages/faultforge`
+ `packages/faultforge_cli` uv workspace, with public top-level modules
(`faultforge.encoding`, `faultforge.experiment`, `faultforge.dataset`,
`faultforge.loading`, `faultforge.progress`, `faultforge.fingerprint`). The Rust
side moved from one crate to a `picker`/`memory`/`bindings` workspace.

### Added

- **`Fingerprint`**: a structural, serializable identity for an experiment's
  configuration (model, dataset, encoder, fault count, etc). Saved results now
  carry a fingerprint, and resuming/verifying a saved file against your current
  setup raises a precise `FingerprintError` (e.g.
  `encoder.head[0].bits_per_chunk: changed parameter`) instead of silently
  mixing incompatible data - 0.1 had no protection against this at all.
  Fingerprint scalars are normalized (e.g. `encoded_memory`'s fault count is
  always recorded as a resolved integer, whether you originally passed
  `--faults` or `--bit-error-rate`), so switching between equivalent ways of
  specifying the same config across a resume doesn't spuriously trip
  verification.
- **Stop conditions** (`StopCondition`, `MaxRuns`, `AdditionalRuns`,
  `Stability`): experiments can now stop themselves once a statistical stability
  threshold (95% CI margin of error, as a percentage of the mean) is reached,
  instead of always running a fixed, manually chosen number of iterations.
  Ctrl+C is itself a stop condition, so interrupting an experiment finishes the
  in-flight run cleanly instead of corrupting state.
- **Atomic, resumable saving** (`SaveConfig`, `Experiment.save_atomic`):
  periodic autosave during a long run, written via a temp file + rename so a
  crash mid-write can't corrupt the output file.
- **zstd-compressed save files**: `save`/`save_atomic`/`load_from` and the CLI's
  `--output` can transparently read/write zstd-compressed result files (detected
  by sniffing the file's magic bytes, not its extension); the new `--compress`
  flag only affects newly created files - an existing output file keeps whatever
  format it already had.
- **Saved-result inspection and visualization** for `encoded_memory`: a saved
  result file is now a public `SavedResult` type with `.load()`/`.scores()`/
  `.reliability_metric()`/`.bit_error_rate()`, so it can be inspected without
  reconstructing the model/dataset that produced it. The CLI's new `compare` and
  `heatmap` commands build on this to plot accuracy/reliability across a batch
  of saved runs, grouped by matching `Fingerprint`s - a from-scratch redesign of
  0.1's scatter/mean/configurations commands, which were dropped early in the
  Fingerprint rewrite and are now properly reinstated (plotting stays a
  `faultforge_cli`-only dependency; the `faultforge` library itself doesn't
  depend on matplotlib).
- **`StuckAt` faults**: alongside bit flips (`BitFlip`), faults can now be
  "stuck at 0/1" - a fault model 0.1 didn't support at all.
- **Resumable `Picker`**: the Fisher-Yates fault-location sampler can now
  reconstruct its state from a set of already-returned indices
  (`Picker.from_returned`) and is exposed directly to Python, so a long
  fault-injection campaign can be suspended and resumed without replaying
  already-picked locations.
- **Fault injection API reshaped around explicit, batchable `Fault` values**:
  0.1 exposed one function, `tensor_list_fault_injection(ts, faults_count)`,
  which always flipped `faults_count` random bits chosen internally by Rust - no
  control over which bits, and no non-flip fault types. 0.2 splits this into
  `Picker` (choose locations explicitly, see above) plus
  `tensor_list_fault`/`tensor_list_faults(ts, [(fault, target_bit), ...])`,
  applying one or many explicit `Fault`s (`BitFlip` or `StuckAt`) at
  caller-chosen bit positions in a single call (`Encoding.apply_faults`/
  `EncodedModule.apply_faults` do the same for encoded memory). A batched call
  still pays the tensor round-trip cost once for the whole set, same as 0.1's
  count-based calls did; what's newly improved is non-uniform/encoded sequences
  (MSET/CEP), which used to locate each fault's owning chunk with an O(n) linear
  scan and now do it in O(log n).
- **Per-run bitwise comparison and fault-injection summaries** in the
  `encoded_memory` experiment: `--compare-bitwise` records per-run XOR bitmasks
  against the golden model (with `discard-bitmasks`/ `discard_bitmasks_in_file`
  to shrink a saved result afterwards once the raw bitmasks aren't needed), and
  `--fault-summary` reports a histogram of bits flipped, parameters affected,
  and the fraction of faults masked by the encoding.
- **`Progress` reporting** (`faultforge.progress`): long operations (model
  download, dataset precompute, encoding, fault injection) now emit periodic
  "N/total (%) - elapsed" log messages instead of running silently.
- **`CachedDataset`/`precompute()`**: keeps all batches of a dataset in memory
  once, instead of re-loading/transforming them on every fault-injection run.
- A dedicated CLI package, `faultforge_cli` (`faultforge-cli encoded-memory
  record ...`), covering model/dataset setup, fault injection, encoding scheme
  selection, and recording/stop-condition flags - see `--help` (now also
  available as `-h`).

### Changed

- **`System` → `Experiment` + `ModelBundle` + `EncodedModule`.** 0.1's `System`
  was a single class mixing model loading, dataset loading, accuracy
  computation, and fault injection, with one subclass per domain (`CifarSystem`,
  `ImageNetSystem`, `EncodedSystem`). 0.2 splits these concerns: `ModelBundle`
  (`Cifar`, `ImageNet`) only knows how to load a model + its eval dataset;
  `EncodedModule` wraps any `torch.nn.Module` so its parameters live in
  simulated encoded memory; `Experiment` is a pure behavior ABC
  (`run`/`scores`/`serialize`/`deserialize`) with all the
  run-loop/stopping/saving machinery described above generic and reusable across
  experiment types. `faultforge.experiments.encoded_memory` is the one built-in
  experiment and a reference implementation for new ones.
- **Encoding schemes renamed and reorganized**, same underlying ideas: 0.1's
  "embedded parity" is now **CEP** (Chunked Embedded Parity), 0.1's "MSB"
  (odd-duplicate-bits + majority vote) is now **MSET** (Most Significant
  Exponent bit Triplication, generalized over chunk count), SECDED is
  unchanged in concept. All three plus `sequence` (compose encoders) and
  `identity` (unprotected baseline) now share common `Encoder`/`Encoding`/
  `TensorEncoder` ABCs with their own fingerprints.
- **Rust workspace restructured**: one `faultforge` crate +
  `faultforge-bindings` → three crates, `picker` (fault-location sampling),
  `memory` (bit buffers + ECC encodings), `bindings` (PyO3 glue).
- **Repo layout**: single Python package → `uv` workspace
  (`packages/faultforge`, `packages/faultforge_cli`); all dependencies
  (`matplotlib`, `pillow`, `timm`, `torch`, `torchvision`, `scipy`, ...) are now
  required by default rather than gated behind `all`/`cifar`/`imagenet`/`cli`
  extras.
- Dev tooling switched from `basedpyright` to `ty`.
- **`strip_faults` CLI command** - Now available as `discard-bitmasks`.

### Removed

- **Bit-pattern encoding** (0.1's `BitPattern`/`encoding_bit_pattern` bindings)
  - no equivalent in the new `memory` crate. This was mostly an experimental
  technique which wasn't as useful as initially hoped, Included in v0.1 for
  completeness sake but now removed to get rid of the maintenacne burden.
- **Standalone bitwise-comparison Rust bindings**
  (`compare_array_list_bitwise_*`) - comparison logic now lives on the
  Python side.
