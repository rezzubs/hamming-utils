# Systolic

A [FaultForge](https://github.com/rezzubs/faultforge) experiment: fault
injection into a weight-stationary systolic array evaluating a model's
`nn.Linear`/`nn.Conv2d` layers. Installing it provides the
`SystolicFaultInjection` library API, plus the `SystolicBackend` abstraction
it's built on (`TorchBackend` as the fault-free golden path, `SimulatedBackend`
as the cycle-accurate oracle, `LiftedBackend` as the fast torch-side
workhorse).

`SystolicFaultInjection` injects a single stuck-at register fault (targeting
the activation, weight, or accumulator register of one processing element)
per run, and scores the result according to a `ReliabilityMetric`:
`Accuracy`, `AccuracyDegradation`, `Sdc` (Silent Data Corruption), or
`Top1Sdc` (Critical SDC).

No CLI exists for this experiment yet; use it as a library, following
`encoded_memory`'s `experiment.py` as the reference `Experiment`
implementation.

## Installation

This experiment isn't published to PyPI but doesn't depend on any native code
other than what `faultforge` provides. `faultforge` itself is a pinned,
regular dependency so it installs straight from its prebuilt PyPI wheel.
Point pip at this subdirectory of whichever repository revision you want:

```sh
# development (main is kept in sync with the newest code going forward)
pip install 'systolic @ git+https://github.com/rezzubs/faultforge.git#subdirectory=experiments/systolic'
```

Requires Python 3.14 or newer.

## Library usage

```python
from faultforge.experiment import MaxRuns
from faultforge.loading import Cifar, CifarDataset, CifarModel
from systolic import RegisterFaults, ReliabilityMetric, SystolicFaultInjection

bundle = Cifar(model=CifarModel.ResNet20, dataset=CifarDataset.Cifar10)

experiment = SystolicFaultInjection(
    bundle,
    array=(32, 32),
    fault=RegisterFaults(),
    backend="lifted",
    reliability_metric=ReliabilityMetric.Sdc,
)
experiment.run_loop(stop_conditions=[MaxRuns(total=50)])
experiment.save_atomic("result.json")
```
