from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torchvision.models.densenet import DenseNet
from torchvision.models.resnet import ResNet


class HamNet(torch.nn.Module, ABC):
    def __init__(self, backbone: torch.nn.Module) -> None:
        super().__init__()
        self.backbone, num_features = self.set_backbone(backbone)

        self.meta_net = torch.nn.Sequential(
            torch.nn.Linear(3, 64),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 128),
            torch.nn.SiLU(),
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(num_features + 128, 1024),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1024, 7),
        )

    @abstractmethod
    def set_backbone(self, backbone: torch.nn.Module) -> Tuple[torch.nn.Module, int]:
        pass

    def forward(self, img: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        img_features = self.backbone(img)
        meta_features = self.meta_net(meta)
        x = torch.cat([img_features, meta_features], dim=1)
        return self.classifier(x)


class HamDenseNet(HamNet):
    def __init__(self, densenet: DenseNet) -> None:
        super().__init__(backbone=densenet)

    def set_backbone(self, backbone: torch.nn.Module) -> Tuple[torch.nn.Module, int]:
        num_features = backbone.classifier.in_features
        backbone.classifier = torch.nn.Identity()
        return backbone, num_features


class HamResNet(HamNet):
    def __init__(self, resnet: ResNet) -> None:
        super().__init__(backbone=resnet)

    def set_backbone(self, backbone: torch.nn.Module) -> Tuple[torch.nn.Module, int]:
        num_features = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()
        return backbone, num_features
