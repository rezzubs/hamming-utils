"""Loading ImageNet models and the ImageNet validation dataset."""

import copy
import enum
import typing
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import override

import timm
import torch
import torchvision
from PIL import Image
from torch import (
    Tensor,
    nn,
)
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from faultforge._internal.dataset import BatchedDataset, DeviceLike
from faultforge._internal.fingerprint import Fingerprint
from faultforge._internal.io import AnyPath
from faultforge._internal.loading.abc import DEFAULT_DTYPE, ModelBundle
from faultforge._internal.progress import Progress, stage

type Transform = Callable[[Image.Image], Tensor]


def _get_tim_transform(model: nn.Module) -> Transform:
    """Get a Transform for a Module loaded from timm."""
    from timm.data.config import (
        resolve_data_config,
    )
    from timm.data.transforms_factory import create_transform

    pretrained_cfg = model.pretrained_cfg
    config = resolve_data_config(pretrained_cfg=pretrained_cfg)
    assert isinstance(config, object)

    if not isinstance(config, dict):
        raise TypeError(f"Expected config to be a dict, got {type(config)}")

    transform = create_transform(**config)

    if not isinstance(transform, transforms.Compose):
        raise TypeError(f"Expected transform to be a Compose, got {type(transform)}")

    return typing.cast(Transform, transform)


class ImageNetModel(enum.StrEnum):
    """A pretrained image classifier evaluated on ImageNet."""

    # Hugging Face models
    DeitTiny = "deit_tiny_patch16_224"
    DeitBase = "deit_base_patch16_224"
    SwinTiny = "swin_tiny_patch4_window7_224"
    VitBase = "vit_base_patch16_224"
    VitTiny = "vit_tiny_patch16_224"

    # Torchvision models
    InceptionV3 = "inception_v3"
    MobileNetV2 = "mobilenet_v2"
    ResNet152 = "resnet152"


@dataclass(slots=True)
class ImageNet(ModelBundle):
    """A Description for loading the Imagenet dataset and models.

    The model is cached inside the instance on first load and `load_model`
    returns a copies of the cached model.
    """

    _root: AnyPath
    _kind: ImageNetModel
    _model: nn.Module | None
    """A cached copy of the loaded model.

    Needs to be cached to load the dataset. See _get_tim_transform.
    """

    def __init__(self, kind: ImageNetModel, root: AnyPath):
        """Describe an ImageNet model/dataset pair.

        `root` should point at a local ImageNet directory that stores the files:

        - ILSVRC2012_devkit_t12.tar.gz
        - ILSVRC2012_img_val.tar

        which can be downloaded from
        https://image-net.org/challenges/LSVRC/2012/2012-downloads.php
        """
        self._root = root
        self._kind = kind
        self._model = None

    def _load_model(self, *, progress: Progress | None = None) -> nn.Module:
        with stage(progress, f"Loading model {self._kind.name}"):
            match self._kind:
                case (
                    ImageNetModel.DeitTiny
                    | ImageNetModel.DeitBase
                    | ImageNetModel.SwinTiny
                    | ImageNetModel.VitBase
                    | ImageNetModel.VitTiny
                ):
                    root_module = timm.create_model(self._kind.value, pretrained=True)
                case ImageNetModel.InceptionV3:
                    root_module = torchvision.models.inception_v3(
                        weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1
                    )
                case ImageNetModel.MobileNetV2:
                    root_module = torchvision.models.mobilenet_v2(
                        weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2
                    )
                case ImageNetModel.ResNet152:
                    root_module = torchvision.models.resnet152(
                        weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V2
                    )

        return root_module

    def _cached_model(self, *, progress: Progress | None = None) -> nn.Module:
        if self._model is None:
            self._model = self._load_model(progress=progress)
        return self._model

    def get_transform(self, *, progress: Progress | None = None) -> Transform:
        """Get the proper preprocessing transform for this model."""

        match self._kind:
            case (
                ImageNetModel.DeitTiny
                | ImageNetModel.DeitBase
                | ImageNetModel.SwinTiny
                | ImageNetModel.VitBase
                | ImageNetModel.VitTiny
            ):
                return _get_tim_transform(self._cached_model(progress=progress))
            case ImageNetModel.InceptionV3:
                weights = torchvision.models.Inception_V3_Weights.IMAGENET1K_V1
            case ImageNetModel.MobileNetV2:
                weights = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2
            case ImageNetModel.ResNet152:
                weights = torchvision.models.ResNet152_Weights.IMAGENET1K_V2

        return weights.transforms()

    @override
    def fingerprint(self) -> Fingerprint:
        return Fingerprint(kind="imagenet", scalars={"model": self._kind.value})

    @override
    def load_model(
        self,
        device: DeviceLike,
        *,
        dtype: torch.dtype = DEFAULT_DTYPE,
        progress: Progress | None = None,
    ) -> nn.Module:
        return copy.deepcopy(self._cached_model(progress=progress)).to(
            device=device, dtype=dtype
        )

    @override
    def load_dataset(
        self, batch_size: int, device: DeviceLike, *, progress: Progress | None = None
    ) -> BatchedDataset:
        with stage(progress, "Loading ImageNet dataset"):
            dataset = datasets.ImageNet(
                Path(self._root),
                split="val",
                transform=self.get_transform(progress=progress),
            )
            assert isinstance(dataset, Dataset)
        return BatchedDataset.from_dataset(dataset, batch_size, device)
