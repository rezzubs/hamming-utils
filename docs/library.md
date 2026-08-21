# Library usage

This is a tutorial-oriented tour of FaultForge's building blocks and how they
compose. See [`experiments/encoded_memory`'s README](../experiments/encoded_memory/README.md)
for the reference experiment that ties all of this together, or the
`faultforge` package's module docstrings for API-level detail.

## `ModelBundle`

A [`ModelBundle`](../src/faultforge/loading.py) knows how
to load a model together with its evaluation dataset. It's the thing an
experiment asks for a model and data - it doesn't know anything about
encoding or fault injection itself.

```python
from faultforge.loading import Cifar, CifarDataset, CifarModel

bundle = Cifar(model=CifarModel.ResNet20, dataset=CifarDataset.Cifar10)
```

`Cifar` and `ImageNet` are the two built-in bundles (`faultforge.loading`).
To support a new model source, subclass `ModelBundle` and implement:

- `load_model(device, *, dtype, progress) -> nn.Module`
- `load_dataset(batch_size, device, *, progress) -> BatchedDataset`
- `fingerprint() -> Fingerprint` - a structural identity for *what* is being
  loaded (model name, dataset variant, ...), deliberately excluding
  environmental details like device or filesystem paths.

The `encoded_memory` experiment's test suite has a `_FakeBundle`
(`experiments/encoded_memory/tests/conftest.py`) as a minimal example: it
wraps a plain `nn.Linear` and a random `TensorDataset`, useful as a template
when writing your own.

## `Encoder` / `Encoding`

An `Encoder` (`faultforge.encoding`) turns a list of tensors into an
`Encoding` - a protected representation that can be decoded back and that
knows how to have faults applied to it at specific bit indices:

```python
from faultforge.encoding import SecdedEncoder

encoder = SecdedEncoder(bits_per_chunk=64)
encoding = encoder.encode(tensors)

encoding.apply_fault(fault, target_bit)  # or apply_faults(...) for a batch
recovered = encoding.decode()
```

`TensorEncoder`/`TensorEncoding` are a refinement for encodings whose
underlying storage is itself a list of tensors, which makes them
composable: `EncoderSequence` chains several such encoders together (e.g. an
MSET or CEP pass followed by a SECDED pass over the result). Encoders that
transform each tensor in place (`CepEncoder`, `MsetEncoder`) build on a
further `InPlaceEncoder`/`InPlaceEncoding` convenience base.

`IdentityEncoder` stores tensors unmodified - the unprotected baseline to
compare other encodings against.

### `EncodedModule`

`EncodedModule` (`faultforge.encoding`) wraps a `torch.nn.Module` so its
parameters live in simulated encoded memory instead of as plain tensors:

```python
from faultforge.encoding import EncodedModule

encoded = EncodedModule(model, encoder)
encoded.apply_faults(faults)
decoded_model = encoded.decode()  # a copy of `model` with decoded parameters
```

This is the piece the `encoded_memory` experiment builds its fault injection
around - see
[`experiments/encoded_memory`'s README](../experiments/encoded_memory/README.md#library-usage).

## Fault injection primitives

A fault is either a bit flip or a stuck-at value:

```python
from faultforge import BitFlip, StuckAt

flip = BitFlip()
stuck_low = StuckAt.Zero
```

`Picker` is a Fisher-Yates-based sampler for choosing fault locations without
repeats; it can resume from a set of already-returned indices
(`Picker.from_returned`), so a long fault-injection campaign can be
suspended and picked back up without replaying earlier picks.

Prefer batched application - `tensor_list_faults`, or
`Encoding.apply_faults`/`EncodedModule.apply_faults` - over looping a single
fault at a time: a batch pays its tensor conversion overhead once for the
whole set instead of once per fault.

## `Experiment`

`Experiment` (`faultforge.experiment`) is the base class for a repeatable,
scoreable unit of work. A subclass implements:

- `run() -> None` - perform one iteration and record it internally.
- `scores() -> Sequence[float]` - every recorded score so far, in run order.
  This is the only view the generic machinery needs.
- `serialize() -> str` / `deserialize(content: str) -> None` - turn state
  into a string and back. Include your own `Fingerprint` in the output and
  check it via `Fingerprint.raise_if_differs` if you want a loaded file
  verified against the current configuration.

In exchange, the base class provides:

- `run_loop(*, stop_conditions=(), save_config=None)` - repeatedly calls
  `run()` until a stop condition fires (including Ctrl+C, handled cleanly so
  an in-flight run finishes before stopping), printing a status line each
  iteration and optionally autosaving.
- `mean_score()` / `margin_of_error()` - a 95% confidence interval over
  `scores()`.
- `save`/`save_atomic` and `load_from` - atomic (temp file + rename), so a
  crash mid-write can't corrupt the output file; both can transparently
  read/write zstd-compressed files (`compressed=True`, detected on load by
  sniffing the file's magic bytes, not its extension).

Stop conditions (`faultforge.experiment`) are plain callables you pass to
`run_loop`:

- `Stability(min_samples, threshold)` - stop once the mean's relative 95%
  margin of error falls below `threshold` percent.
- `MaxRuns(total)` - stop once the total run count (including any loaded
  from a save file) reaches `total`.
- `AdditionalRuns(count)` - stop after `count` more runs on top of however
  many already existed when first checked.

```python
from faultforge.experiment import SaveConfig, Stability

experiment.run_loop(
    stop_conditions=[Stability(min_samples=10, threshold=1.0)],
    save_config=SaveConfig(path="result.json.zst", interval_seconds=30, compressed=True),
)
```

## `Fingerprint`

A `Fingerprint` (`faultforge.Fingerprint`) is a small, structural,
serializable identity for an experiment component - an encoder, a model
bundle, an experiment's own configuration. It captures only the semantics
that change what the component does (not environmental details like device
or filesystem paths), and composite components nest their parts' fingerprints
under `children`.

Comparing two fingerprints (`Fingerprint.diff`/`raise_if_differs`) produces
path-qualified differences, e.g. `encoder.head[0].bits_per_chunk: changed
parameter`, so a mismatch between a saved result and your current setup
fails with a precise reason instead of silently mixing incompatible data.
`faultforge.fingerprint` has diffing helpers (`FingerprintError`,
`format_differences`) for presenting these to a user.

## Putting it together

```python
from encoded_memory import (
    EncodedFaultInjection,
    ReliabilityMetric,
)
from faultforge.encoding import SecdedEncoder
from faultforge.experiment import MaxRuns, SaveConfig, Stability
from faultforge.loading import Cifar, CifarDataset, CifarModel

bundle = Cifar(model=CifarModel.ResNet20, dataset=CifarDataset.Cifar10)
encoder = SecdedEncoder(bits_per_chunk=64)

experiment = EncodedFaultInjection(
    bundle,
    encoder,
    ReliabilityMetric.Accuracy,
    faults=1e-3,
)

save_config = SaveConfig(path="result.json", interval_seconds=30)
try:
    experiment.load_from(save_config.path)
except FileNotFoundError:
    pass

experiment.run_loop(
    stop_conditions=[
        Stability(min_samples=10, threshold=1.0),
        MaxRuns(total=200),
    ],
    save_config=save_config,
)
```

`EncodedFaultInjection` is the reference `Experiment` implementation to
follow when writing your own - see
[`experiments/encoded_memory`'s README](../experiments/encoded_memory/README.md)
for how it uses everything above.
