"""Layers whose matmul is routed through a SystolicBackend."""

from typing import final, override

import torch.nn.functional as F
from faultforge.progress import Progress, stage
from torch import Tensor, nn

from systolic.backend import SystolicBackend


def conv2d_output_height(conv2d: nn.Conv2d, input_height: int) -> int:
    """The output height produced by `conv2d` for an input of `input_height`.

    Standard convolution output-size formula: pad the input, subtract the
    kernel's footprint once dilation has spread it out, then see how many
    times the stride fits.
    """
    if isinstance(conv2d.padding, str):
        raise ValueError("string padding is not supported")
    padding_height = 2 * conv2d.padding[0]
    effective_input_height = padding_height + input_height
    kernel_height = conv2d.kernel_size[0]
    dilation_height = conv2d.dilation[0]
    effective_kernel_height = dilation_height * (kernel_height - 1) + 1
    stride_height = conv2d.stride[0]
    return (effective_input_height - effective_kernel_height) // stride_height + 1


def conv2d_output_width(conv2d: nn.Conv2d, input_width: int) -> int:
    """The output width produced by `conv2d` for an input of `input_width`.

    See `conv2d_output_height` for the formula; this is the same computation
    along the width axis.
    """
    if isinstance(conv2d.padding, str):
        raise ValueError("string padding is not supported")
    padding_width = 2 * conv2d.padding[1]
    effective_input_width = padding_width + input_width
    kernel_width = conv2d.kernel_size[1]
    dilation_width = conv2d.dilation[1]
    effective_kernel_width = dilation_width * (kernel_width - 1) + 1
    stride_width = conv2d.stride[1]
    return (effective_input_width - effective_kernel_width) // stride_width + 1


def _layer_label(name: str, position: str, kind: str, shape: str) -> str:
    """Build a per-layer progress label: `"features.7 [8/20] Conv2d(576x3136x8)"`.

    `name` is the dotted path within the wrapped model and `position` its place
    among the model's mapped layers, both empty for a layer built outside a
    `BackendModel`. `shape` is the backend matmul's `out x in x batch`
    dimensions, the number that actually predicts how slow the layer will be.

    `position` reflects where a layer sits in the module tree, not how far a
    forward pass has got: a model whose `forward` visits its children out of
    registration order will show them out of order.
    """
    parts = [part for part in (name, f"[{position}]" if position else "") if part]
    prefix = f"{' '.join(parts)} " if parts else ""
    return f"{prefix}{kind}({shape})"


@final
class MappedLinear(nn.Module):
    """Routes an `nn.Linear`'s matmul through a `SystolicBackend`."""

    # Deliberately not a @dataclass: nn.Module.__setattr__ needs
    # super().__init__() to run before any attribute assignment, which a
    # dataclass-generated __init__ has no hook to do, and dataclass's auto
    # __eq__/__repr__ would conflict with nn.Module's own (tensor comparisons in
    # __eq__, tree-printing repr).
    def __init__(
        self,
        inner: nn.Linear,
        backend: SystolicBackend,
        *,
        name: str = "",
        position: str = "",
        progress: Progress | None = None,
    ) -> None:
        super().__init__()
        self.inner = inner
        self.backend = backend
        self.name = name
        self.position = position
        self.progress = progress

    @override
    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 2:
            raise ValueError("MappedLinear only supports 2d inputs")
        out_features, in_features = self.inner.weight.shape
        label = _layer_label(
            self.name,
            self.position,
            "Linear",
            f"{out_features}x{in_features}x{x.shape[0]}",
        )
        with stage(self.progress, label):
            # nn.Linear.weight is already (out_features, in_features); only the
            # activations need transposing to the backend's (in_features, batch)
            # convention, then the result transposed back to (batch, out_features).
            out = self.backend.matmul(self.inner.weight, x.T).T
            if self.inner.bias is not None:  # None when nn.Linear(..., bias=False)
                out = out + self.inner.bias
        return out


@final
class MappedConv2d(nn.Module):
    """Routes an `nn.Conv2d`'s matmul through a `SystolicBackend` via im2col.

    `im2col` ("image to columns") turns a convolution into a single matrix
    multiplication. Every position the kernel visits gets its own column
    containing the flattened values under the kernel at that position and the
    kernel's weights get flattened into a matrix. One matmul of that weight
    matrix against the matrix of columns then computes every output value at
    once. `forward` below builds that matmul step by step, with each tensor's
    shape spelled out in a comment as it changes.

    Only `groups=1` convolutions are supported. A grouped convolution is a
    block-diagonal operation (independent per-group matmuls), which the
    current systolic array model has no representation for.
    """

    # See comment on MappedLinear `__init__`.
    def __init__(
        self,
        inner: nn.Conv2d,
        backend: SystolicBackend,
        *,
        name: str = "",
        position: str = "",
        progress: Progress | None = None,
    ) -> None:
        super().__init__()
        if inner.groups != 1:
            raise ValueError(
                f"MappedConv2d does not support grouped convolutions "
                f"(got groups={inner.groups}); only groups=1 is currently supported"
            )
        self.inner = inner
        self.backend = backend
        self.name = name
        self.position = position
        self.progress = progress

    @override
    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected a 4d input, got {x.dim()}d")
        if isinstance(self.inner.padding, str):
            raise ValueError("string padding is not supported by F.unfold")

        # x: (batch_size, input_channels, input_height, input_width)
        batch_size = x.shape[0]
        input_channels = self.inner.in_channels
        output_channels = self.inner.out_channels
        kernel_height, kernel_width = self.inner.kernel_size

        output_height = conv2d_output_height(self.inner, x.shape[2])
        output_width = conv2d_output_width(self.inner, x.shape[3])
        # The kernel visits one position per output pixel.
        output_position_count = output_height * output_width
        # How many numbers the kernel reads at each position it visits.
        flattened_patch_size = input_channels * kernel_height * kernel_width

        label = _layer_label(
            self.name,
            self.position,
            "Conv2d",
            f"{output_channels}x{flattened_patch_size}x{batch_size * output_position_count}",
        )
        with stage(self.progress, label):
            # im2col: for every position the kernel visits, extract the
            # flattened_patch_size values under it into one column, for every
            # batch element at once.
            #
            # patches: (batch_size, flattened_patch_size, number_of_output_positions)
            patches = F.unfold(
                x,
                self.inner.kernel_size,
                dilation=self.inner.dilation,
                padding=self.inner.padding,
                stride=self.inner.stride,
            )

            # Flatten the kernel's weights into a matrix that, multiplied by one
            # patch column, produces that position's output value across all
            # output channels at once.
            #
            # weight_matrix: (output_channels, flattened_patch_size)
            weight_matrix = self.inner.weight.reshape(
                output_channels, flattened_patch_size
            )

            # The backend's matmul only has one "batch"/"free" axis - weights
            # (out_features, in_features) @ activations (in_features, batch) ->
            # (out_features, batch)). To cover every batch element with a single
            # matmul call instead of one call per batch element, lay every batch
            # element's patch columns side by side along that axis.
            #
            # Tiling batches together only needs to preserve the values a
            # plain matmul would produce, which it does: the zero-padding
            # cycles added by the array's dataflow (`shift_activations` in
            # `crates/systolic/src/array.rs`) never contribute to a real output
            # under plain multiply-accumulate. Whether batching also preserves
            # fault behavior is a separate question, answered on `XorMaskHook`'s doc
            # comment for the hooks implemented today.
            #
            # patches_matrix: (flattened_patch_size, batch_size * number_of_output_positions)
            patches_matrix = patches.permute(1, 0, 2).reshape(
                flattened_patch_size, batch_size * output_position_count
            )

            # One matmul computes every output value, for every output channel,
            # for every batch element, all at once.
            #
            # raw_output: (output_channels, batch_size * number_of_output_positions)
            raw_output = self.backend.matmul(weight_matrix, patches_matrix)

            # Undo the batch-tiling from above, then move the batch axis back to
            # the front to match PyTorch's (batch, channel, height, width)
            # convention.
            #
            # output: (batch_size, output_channels, number_of_output_positions)
            output = raw_output.reshape(
                output_channels, batch_size, output_position_count
            ).permute(1, 0, 2)

            # Unflatten number_of_output_positions back into a 2d
            # (output_height, output_width) grid.
            #
            # out: (batch_size, output_channels, output_height, output_width)
            out = output.reshape(
                batch_size, output_channels, output_height, output_width
            )

            if self.inner.bias is not None:
                # Differs from MappedLinear: broadcasts over
                # (batch, channel, height, width), so it needs an explicit
                # (1, output_channels, 1, 1) reshape rather than relying on
                # trailing-dim broadcast.
                out = out + self.inner.bias.view(1, -1, 1, 1)
        return out
