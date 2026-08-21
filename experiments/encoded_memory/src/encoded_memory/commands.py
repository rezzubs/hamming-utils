"""The `encoded-memory` CLI commands (recording and plotting)."""

import enum
import logging
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import torch
import typer
from matplotlib.backends.registry import BackendFilter, backend_registry
from matplotlib.figure import Figure
from faultforge.dataset import DEFAULT_BATCH_SIZE
from faultforge.encoding import (
    CepEncoder,
    CepScheme,
    Encoder,
    EncoderSequence,
    IdentityEncoder,
    MsetEncoder,
    SecdedEncoder,
)
from faultforge.experiment import (
    AdditionalRuns,
    MaxRuns,
    SaveConfig,
    Stability,
    StopCondition,
)
from encoded_memory import (
    DetailedResult,
    EncodedFaultInjection,
    ReliabilityMetric,
    discard_bitmasks_in_file,
)
from faultforge.fingerprint import FingerprintError
from faultforge.io import is_compressed
from faultforge.loading import (
    Cifar,
    CifarDataset,
    CifarModel,
    ImageNet,
    ImageNetModel,
    ModelBundle,
)
from faultforge.progress import Progress
from encoded_memory.plots import (
    GroupBy,
    build_compare_figure,
    build_heatmap_figure,
)
from encoded_memory.results import (
    build_configurations,
    discover_result_files,
    load_results,
)

app = typer.Typer(
    pretty_exceptions_enable=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)
logger = logging.getLogger(__name__)


class DatasetChoice(enum.StrEnum):
    Cifar10 = "cifar10"
    Cifar100 = "cifar100"
    ImageNet = "imagenet"


def _init_model_bundle(
    dataset: DatasetChoice,
    model: str | None,
    imagenet_root: str | None,
    batch_size: int,
    preload_batches: bool,
    device: str,
) -> ModelBundle:
    """Build the `ModelBundle` for the given CLI choices and load the model/dataset from it."""
    if model is None:
        raise typer.BadParameter("A --model must be specified.", param_hint="--model")

    bundle: ModelBundle
    match dataset:
        case DatasetChoice.Cifar10 | DatasetChoice.Cifar100:
            try:
                cifar_model = CifarModel(model)
            except ValueError as error:
                choices = ", ".join(m.value for m in CifarModel)
                raise typer.BadParameter(
                    f"Unknown model {model!r} for dataset {dataset.value}. Choices: {choices}",
                    param_hint="--model",
                ) from error
            bundle = Cifar(model=cifar_model, dataset=CifarDataset(dataset.value))
        case DatasetChoice.ImageNet:
            if imagenet_root is None:
                raise typer.BadParameter(
                    "--imagenet-root is required when --dataset imagenet.",
                    param_hint="--imagenet-root",
                )
            try:
                imagenet_model = ImageNetModel(model)
            except ValueError as error:
                choices = ", ".join(m.value for m in ImageNetModel)
                raise typer.BadParameter(
                    f"Unknown model {model!r} for dataset imagenet. Choices: {choices}",
                    param_hint="--model",
                ) from error
            bundle = ImageNet(kind=imagenet_model, root=imagenet_root)

    return bundle


def _resolve_encoder(
    *, mset: bool, cep: bool, cep_scheme: CepScheme, secded: int | None
) -> Encoder:
    match (mset, cep):
        case (True, False):
            head = MsetEncoder()
        case (False, True):
            head = CepEncoder(cep_scheme)
        case (True, True):
            raise typer.BadParameter("Using MSET and CEP together is not allowed.")
        case (False, False):
            head = None

    match (head, secded):
        case (None, None):
            return IdentityEncoder()
        case (_, None):
            # Asserts to help out the type checker.
            assert head is not None
            return head
        case (None, _):
            assert secded is not None
            return SecdedEncoder(secded)
        case (_, _):
            assert secded is not None
            assert head is not None
            return EncoderSequence([head], SecdedEncoder(secded))


@app.command()
def list_models(
    dataset: Annotated[
        DatasetChoice, typer.Option(help="Which dataset to use")
    ] = DatasetChoice.ImageNet,
) -> None:
    """List all available models for the given dataset."""
    match dataset:
        case DatasetChoice.Cifar10 | DatasetChoice.Cifar100:
            models = [model.value for model in CifarModel]
        case DatasetChoice.ImageNet:
            models = [model.value for model in ImageNetModel]

    for model in models:
        typer.echo(model)


@app.command(
    no_args_is_help=True,
)
def record(
    model: Annotated[
        str,
        typer.Option(
            help="Which model to use. Choices depend on the dataset. The the list-models command can be used to see available models.",
            rich_help_panel="Model Setup",
        ),
    ],
    dataset: Annotated[
        DatasetChoice,
        typer.Option(
            help="Which dataset to use",
            rich_help_panel="Model Setup",
        ),
    ] = DatasetChoice.ImageNet,
    imagenet_root: Annotated[
        str | None,
        typer.Option(
            help="Path to a local directory containing ILSVRC2012_devkit_t12.tar.gz "
            "and ILSVRC2012_img_val.tar. Required when --dataset is imagenet.",
            rich_help_panel="Model Setup",
        ),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option(
            help="The batch size of the dataset",
            rich_help_panel="Model Setup",
        ),
    ] = DEFAULT_BATCH_SIZE,
    preload_batches: Annotated[
        bool,
        typer.Option(
            help="Preload all batches into memory before starting the experiment. Otherwise, batches are loaded from disk on-demand. This should be set if there is enough memory as it is much faster.",
            rich_help_panel="Model Setup",
        ),
    ] = True,
    batch_limit: Annotated[
        int | None,
        typer.Option(
            help="Use only the first N batches of the dataset",
            rich_help_panel="Model Setup",
        ),
    ] = None,
    f16: Annotated[
        bool,
        typer.Option(
            help="Load and run the model in float16 (half precision) instead of "
            "float32. Halves the encoded bit-width per parameter, changing what "
            "--faults/--bit-error-rate mean numerically. Some PyTorch CPU "
            "kernels have incomplete float16 support for certain architectures; "
            "if you hit a 'not implemented for Half' error, try a CUDA "
            "--device instead.",
            rich_help_panel="Model Setup",
        ),
    ] = False,
    reliability_metric: Annotated[
        ReliabilityMetric,
        typer.Option(
            help="Which metric to use for reliability measurements",
            rich_help_panel="Model Setup",
        ),
    ] = ReliabilityMetric.Accuracy,
    bit_error_rate: Annotated[
        float | None,
        typer.Option(
            "--bit-error-rate",
            "--ber",
            min=0.0,
            max=1.0,
            help="The bit error rate to use. Mutually exclusive with --faults.",
            rich_help_panel="Fault Injection",
        ),
    ] = None,
    faults: Annotated[
        int | None,
        typer.Option(
            min=0,
            help="The number of faults to inject. Mutually exclusive with --bit-error-rate.",
            rich_help_panel="Fault Injection",
        ),
    ] = None,
    compare_bitwise: Annotated[
        bool,
        typer.Option(
            help="Additionally record a bitwise comparison of each run's faulty "
            "parameters against the golden model's, keeping the nonzero xor "
            "results. Increases the size of recorded results.",
            rich_help_panel="Fault Injection",
        ),
    ] = False,
    fault_summary: Annotated[
        bool,
        typer.Option(
            help="Print a per-run fault-injection summary (bits flipped, BER, and "
            "- if --compare-bitwise is also set - affected/masked counts and a "
            "faulty-bit histogram) after each run.",
            rich_help_panel="Fault Injection",
        ),
    ] = False,
    secded: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Use a SECDED encoding with n data bits. Multiples of 8 perform better.",
            rich_help_panel="Encoding Settings",
        ),
    ] = None,
    mset: Annotated[
        bool,
        typer.Option(
            help="Use an MSET encoding. Chunking is determined by the data type.",
            rich_help_panel="Encoding Settings",
        ),
    ] = False,
    cep: Annotated[
        bool,
        typer.Option(
            help="Use a CEP encoding.",
            rich_help_panel="Encoding Settings",
        ),
    ] = False,
    cep_scheme: Annotated[
        CepScheme,
        typer.Option(
            help="Determines the chunking for CEP encoding. Maps to the number of data bits per parity bit in a chunk.",
            rich_help_panel="Encoding Settings",
        ),
    ] = CepScheme.D3P1,
    golden_is_encoded: Annotated[
        bool,
        typer.Option(
            help="Use the non-faulty encoded model as the golden model. By default the unencoded model is used.",
            rich_help_panel="Encoding Settings",
        ),
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option(
            help="A file path to save results to.",
            rich_help_panel="Recording Settings",
        ),
    ] = None,
    autosave: Annotated[
        float | None,
        typer.Option(
            help="Save after N seconds have passed. Ignored if --output is not set. Only saves at the end of the experiment or when interrupted by default.",
            rich_help_panel="Recording Settings",
        ),
    ] = None,
    compress: Annotated[
        bool,
        typer.Option(
            help="Save --output zstd-compressed. Only controls the format of a "
            "newly created file; if --output already exists, its existing "
            "on-disk format (compressed or not) is kept regardless of this flag.",
            rich_help_panel="Recording Settings",
        ),
    ] = False,
    overwrite: Annotated[
        bool,
        typer.Option(
            help="If --output already exists but was recorded with a different configuration, discard it and start fresh instead of aborting.",
            rich_help_panel="Recording Settings",
        ),
    ] = False,
    runs: Annotated[
        int | None,
        typer.Option(
            help="Run the experiment N additional times on top of any existing results (e.g. loaded via --output). Incompatible with --min-runs and --stability-threshold. Can be combined with --max-runs.",
            rich_help_panel="Recording Settings",
        ),
    ] = None,
    max_runs: Annotated[
        int | None,
        typer.Option(
            help="Stop once the results contain N runs in total, including any already loaded via --output. Can be combined with --runs or --stability-threshold.",
            rich_help_panel="Recording Settings",
        ),
    ] = None,
    min_runs: Annotated[
        int | None,
        typer.Option(
            help="Make sure the results contain at least N runs.",
            rich_help_panel="Recording Settings",
        ),
    ] = None,
    stability_threshold: Annotated[
        float | None,
        typer.Option(
            min=0.0,
            max=100.0,
            help="Run until the mean has a margin of error smaller or equal to N% of the mean value at 95% confidence.",
            rich_help_panel="Recording Settings",
        ),
    ] = None,
    device: Annotated[
        str,
        typer.Option(
            help="Which device to use. PyTorch device string.",
            rich_help_panel="Misc Settings",
        ),
    ] = "cpu",
) -> None:
    """Run an encoded memory fault injection experiment and record the results."""
    bundle = _init_model_bundle(
        dataset, model, imagenet_root, batch_size, preload_batches, device
    )

    encoder = _resolve_encoder(
        mset=mset,
        cep=cep,
        cep_scheme=cep_scheme,
        secded=secded,
    )

    if stability_threshold is not None and runs is not None:
        raise typer.BadParameter("Cannot specify both --stability-threshold and --runs")
    if runs is not None and min_runs is not None:
        raise typer.BadParameter("Cannot specify both --runs and --min-runs")

    match (faults, bit_error_rate):
        case (None, None):
            faults_ = 0
        case (None, _):
            assert bit_error_rate is not None
            # Need to make sure it's a float as only floats are treated as BER.
            faults_ = float(bit_error_rate)
        case (_, None):
            assert faults is not None
            faults_ = faults
        case _:
            raise typer.BadParameter(
                "Only one of --faults or --bit-error-rate can be specified"
            )

    dtype = torch.float16 if f16 else torch.float32

    experiment = EncodedFaultInjection(
        bundle,
        encoder,
        reliability_metric,
        golden_is_encoded=golden_is_encoded,
        faults=faults_,
        compare_bitwise=compare_bitwise,
        fault_summary=fault_summary,
        preload_dataset=preload_batches,
        dataset_batch_limit=batch_limit,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        progress=Progress(),
    )
    stop_conditions: list[StopCondition] = []

    save_config: SaveConfig | None = None
    output_exists = False
    if output is not None:
        output = Path(output).expanduser()
        output_exists = output.exists()
        if not output.parent.exists():
            logger.info(f"Creating output parent directory {output.parent}")
            output.parent.mkdir(parents=True)
        else:
            logger.debug(f"Output parent directory {output.parent} already exists")

        effective_compressed = is_compressed(output) if output_exists else compress
        save_config = SaveConfig(
            path=output, interval_seconds=autosave, compressed=effective_compressed
        )

    if stability_threshold is not None:
        if min_runs is None:
            min_samples = 0
        else:
            min_samples = min_runs
        stop_conditions.append(
            Stability(min_samples=min_samples, threshold=stability_threshold)
        )

    if runs is not None:
        stop_conditions.append(AdditionalRuns(runs))

    if max_runs is not None:
        stop_conditions.append(MaxRuns(max_runs))

    if output is not None and output_exists:
        try:
            experiment.load_from(output)
        except FingerprintError as error:
            if not overwrite:
                logger.error(str(error))
                raise typer.Exit(1) from None
            logger.warning(
                f"{output} was recorded with a different configuration and will be overwritten:\n{error}"
            )

    experiment.run_loop(stop_conditions=stop_conditions, save_config=save_config)


@app.command()
def discard_bitmasks(
    path: Annotated[
        Path,
        typer.Argument(
            help="Path to a result file previously recorded with --compare-bitwise."
        ),
    ],
) -> None:
    """Discard any recorded bitmasks from a saved result file, shrinking it.

    Operates directly on the saved file; unlike `record`, this doesn't need to
    reload the model or dataset that produced it.
    """
    discard_bitmasks_in_file(path)


def _split_path_argument(raw: str) -> tuple[Path, str | None]:
    """Splits `raw` on the first literal '=', separating a path from an
    optional legend-label override. Safe since paths never contain '=' on
    this project's Linux-first environment."""
    path_str, sep, label = raw.partition("=")
    return Path(path_str).expanduser(), (label if sep else None)


def _show_or_save(fig: Figure, output: Path | None) -> None:
    if output is not None:
        fig.savefig(output)
        return

    # `plt.show()` silently does nothing on a non-interactive backend (e.g.
    # the "agg" fallback matplotlib picks when no GUI toolkit like PyQt or
    # tkinter is importable) - check first so a missing --output doesn't look
    # like the command did nothing.
    backend = plt.get_backend()
    interactive_backends = {
        name.lower()
        for name in backend_registry.list_builtin(BackendFilter.INTERACTIVE)
    }
    if backend.lower() not in interactive_backends:
        logger.error(
            f"no --output given and matplotlib has no interactive backend "
            f"available (using {backend!r}). Install a GUI toolkit matplotlib "
            "can use (e.g. PyQt6, or a working tkinter), or pass --output to "
            "save the figure to a file instead."
        )
        raise typer.Exit(1)

    # `fig` was built via the `Figure` OO API directly (see `plots`), so
    # pyplot never tracked it the way a `plt.figure()`-created figure would
    # be - `plt.show()` only displays figures pyplot is tracking, so without
    # this it would silently do nothing. Passing an untracked `Figure` as
    # `num` makes `plt.figure()` adopt it instead of creating a new one.
    plt.figure(fig)
    plt.show()


@app.command(no_args_is_help=True)
def compare(
    paths: Annotated[
        list[str],
        typer.Argument(
            help="Result file(s) to compare, or director(ies) recursively "
            "containing them. Results are automatically clustered into "
            "configurations by matching everything in their fingerprint "
            "except the bit error rate, so e.g. several bit error rates of "
            "the same model/encoder/dtype become one line regardless of "
            "which file or directory they came from. Append '=LABEL' to a "
            "path to override the legend label for every file found under "
            "it; otherwise a configuration defaults to the stem of its "
            "first (sorted) file.",
        ),
    ],
    row_by: Annotated[
        GroupBy,
        typer.Option(
            help="Facet the grid into rows by this fingerprint-derived key.",
            rich_help_panel="Grouping",
        ),
    ] = GroupBy.Ungrouped,
    col_by: Annotated[
        GroupBy,
        typer.Option(
            help="Facet the grid into columns by this fingerprint-derived key.",
            rich_help_panel="Grouping",
        ),
    ] = GroupBy.Ungrouped,
    percentile: Annotated[
        float | None,
        typer.Option(
            min=0.0,
            max=100.0,
            help="Plot this percentile of scores per point instead of the mean.",
            rich_help_panel="Display",
        ),
    ] = None,
    log_x: Annotated[
        bool,
        typer.Option(
            "--log-x",
            help="Use a logarithmic x-axis (bit error rate).",
            rich_help_panel="Display",
        ),
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option(
            help="Save the figure to this path instead of opening an "
            "interactive window.",
            rich_help_panel="Output",
        ),
    ] = None,
) -> None:
    """Plot reliability score vs bit error rate, one line per configuration.

    Each PATH is a result file or a directory of them. Everything found is
    pooled and automatically clustered into configurations: results that match
    on every fingerprint field except bit error rate become one line, no matter
    which file or directory they came from. So the usual workflow is to record
    several bit error rates of one setup, then point compare at them to get a
    single line summarizing it.

    Labels: a configuration's legend label defaults to the stem of its first
    file, which is rarely meaningful. Append '=LABEL' to a PATH to name
    everything found under it. The reason to keep each configuration's runs in
    their own directory is so a single 'dir=LABEL' names the whole line at once:

        compare runs/secded=SECDED-8 runs/identity=Identity

    Grid: --row-by / --col-by split the chart into a grid of subplots by a
    fingerprint key (dtype, model, ...). For example --row-by dtype puts f32
    results in the top row and f16 in the next, and the remaining differences
    within a cell become its lines. Omit both for a single chart.
    """
    label_overrides: dict[Path, str] = {}
    source_paths: list[Path] = []
    for raw in paths:
        path, label = _split_path_argument(raw)
        source_paths.append(path)
        if label is not None:
            for file in discover_result_files(path):
                label_overrides[file] = label

    loaded = load_results(source_paths)
    if not loaded:
        logger.error("no result files found")
        raise typer.Exit(1)

    configurations = build_configurations(loaded, label_overrides=label_overrides)

    try:
        fig = build_compare_figure(
            configurations,
            row_by=row_by,
            col_by=col_by,
            percentile=percentile,
            log_x=log_x,
        )
    except ValueError as error:
        logger.error(str(error))
        raise typer.Exit(1) from None

    _show_or_save(fig, output)


@app.command(no_args_is_help=True)
def heatmap(
    paths: Annotated[
        list[Path],
        typer.Argument(
            help="Result file(s) to plot, or director(ies) recursively "
            "containing them. All merged into one figure. Requires results "
            "recorded with --compare-bitwise.",
        ),
    ],
    bins: Annotated[
        int,
        typer.Option(
            min=1,
            help="Number of bins along the score axis.",
            rich_help_panel="Display",
        ),
    ] = 50,
    min_score: Annotated[
        float | None,
        typer.Option(
            help="Discard runs scoring below this value.",
            rich_help_panel="Filtering",
        ),
    ] = None,
    max_score: Annotated[
        float | None,
        typer.Option(
            help="Discard runs scoring above this value.",
            rich_help_panel="Filtering",
        ),
    ] = None,
    max_total_faults: Annotated[
        int | None,
        typer.Option(
            min=0,
            help="Discard runs with more than this many total residual "
            "(post-decode) faults, before decomposing into bit positions.",
            rich_help_panel="Filtering",
        ),
    ] = None,
    skip_multi_bit_faults: Annotated[
        bool,
        typer.Option(
            help="Exclude elements with more than one faulty bit.",
            rich_help_panel="Filtering",
        ),
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option(
            help="Save the figure to this path instead of opening an "
            "interactive window.",
            rich_help_panel="Output",
        ),
    ] = None,
) -> None:
    """Plot per-run score density against faulty bit position."""
    loaded = load_results(paths)
    if not loaded:
        logger.error("no result files found")
        raise typer.Exit(1)

    results = [result for _, result in loaded]
    if not all(isinstance(result.result, DetailedResult) for result in results):
        offenders = [
            str(path)
            for path, result in loaded
            if not isinstance(result.result, DetailedResult)
        ]
        logger.error(
            "heatmap requires results recorded with --compare-bitwise; "
            f"missing bitmasks in: {', '.join(offenders)}"
        )
        raise typer.Exit(1) from None

    try:
        fig = build_heatmap_figure(
            results,
            bins=bins,
            min_score=min_score,
            max_score=max_score,
            max_total_faults=max_total_faults,
            skip_multi_bit_faults=skip_multi_bit_faults,
        )
    except ValueError as error:
        logger.error(str(error))
        raise typer.Exit(1) from None

    _show_or_save(fig, output)
