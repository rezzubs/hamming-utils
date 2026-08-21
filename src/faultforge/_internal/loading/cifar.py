"""Loading CIFAR-10/CIFAR-100 models and datasets."""

import enum
from dataclasses import dataclass
from pathlib import Path
from typing import override

import torch
import torchvision
from torch import nn

from faultforge._internal.dataset import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    BatchedDataset,
    DeviceLike,
)
from faultforge._internal.fingerprint import Fingerprint
from faultforge._internal.loading.abc import DEFAULT_DTYPE, ModelBundle
from faultforge._internal.progress import Progress, stage

CACHE_DIRECTORY = Path("~/.cache/faultforge/").expanduser()


class CifarModel(enum.StrEnum):
    """A model from https://github.com/chenyaofo/pytorch-cifar-models"""

    ResNet20 = "resnet20"
    ResNet32 = "resnet32"
    ResNet56 = "resnet56"
    Vgg11 = "vgg11_bn"
    Vgg13 = "vgg13_bn"
    Vgg16 = "vgg16_bn"
    Vgg19 = "vgg19_bn"
    MobileNetV2_X0_5 = "mobilenetv2_x0_5"
    MobileNetV2_X0_75 = "mobilenetv2_x0_75"
    MobileNetV2_X1_0 = "mobilenetv2_x1_0"
    MobileNetV2_X1_5 = "mobilenetv2_x1_5"
    ShuffleNetV2_X0_5 = "shufflenetv2_x0_5"
    ShuffleNetV2_X1_0 = "shufflenetv2_x1_0"
    ShuffleNetV2_X1_5 = "shufflenetv2_x1_5"
    ShuffleNetV2_X2_0 = "shufflenetv2_x2_0"
    RepVggA0 = "repvgg_a0"
    RepVggA1 = "repvgg_a1"
    RepVggA2 = "repvgg_a2"


class CifarDataset(enum.StrEnum):
    """A CIFAR dataset variant."""

    Cifar10 = "cifar10"
    Cifar100 = "cifar100"

    def _name(self) -> str:
        match self:
            case CifarDataset.Cifar10:
                return "CIFAR10"
            case CifarDataset.Cifar100:
                return "CIFAR100"

    def load(
        self,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: DeviceLike = DEFAULT_DEVICE,
        *,
        progress: Progress | None = None,
    ) -> BatchedDataset:
        """Download (if needed) and load the validation split."""
        with stage(progress, f"Loading dataset {self._name()}"):
            match self:
                case CifarDataset.Cifar10:
                    mean = (0.4913997054, 0.4821583927, 0.4465309978)
                    std = (0.2470322251, 0.2434851378, 0.2615878284)
                    transform = torchvision.transforms.Compose(
                        [
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(mean, std),
                        ]
                    )
                    dataset = torchvision.datasets.CIFAR10(
                        root=CACHE_DIRECTORY,
                        train=False,
                        download=True,
                        transform=transform,
                    )
                case CifarDataset.Cifar100:
                    mean = (0.5070751905, 0.4865489602, 0.4409177899)
                    std = (0.2673342824, 0.2564384639, 0.2761504650)
                    transform = torchvision.transforms.Compose(
                        [
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(mean, std),
                        ]
                    )

                    dataset = torchvision.datasets.CIFAR100(
                        root=CACHE_DIRECTORY,
                        train=False,
                        download=True,
                        transform=transform,
                    )

        return BatchedDataset.from_dataset(dataset, batch_size, device)


@dataclass(slots=True)
class Cifar(ModelBundle):
    """A description for loading a CIFAR model and dataset."""

    model: CifarModel
    dataset: CifarDataset

    @override
    def fingerprint(self) -> Fingerprint:
        return Fingerprint(
            kind="cifar",
            scalars={"model": self.model.value, "dataset": self.dataset.value},
        )

    @override
    def load_model(
        self,
        device: DeviceLike,
        *,
        dtype: torch.dtype = DEFAULT_DTYPE,
        progress: Progress | None = None,
    ) -> nn.Module:
        """Load the model."""

        model_name = f"{self.dataset.value}_{self.model.value}"
        repository = "chenyaofo/pytorch-cifar-models"

        with stage(progress, f"Loading model {self.model.name}"):
            model = torch.hub.load(  # pyright: ignore[reportUnknownMemberType]
                repository,
                model_name,
                pretrained=True,
            )
        if not isinstance(model, nn.Module):
            raise TypeError(
                f"torch.hub.load returned {type(model)}, expected nn.Module"
            )
        return model.to(device=device, dtype=dtype)

    @override
    def load_dataset(
        self, batch_size: int, device: DeviceLike, *, progress: Progress | None = None
    ) -> BatchedDataset:
        """Load the dataset."""
        return self.dataset.load(batch_size, device, progress=progress)
