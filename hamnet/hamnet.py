"""Core model definitions for HamNet and backbone-specific variants.

This module defines the abstract `HamNet` architecture that fuses image
features from a CNN backbone with simple metadata via a small MLP, and
provides concrete implementations for torchvision `DenseNet` and `ResNet`
backbones.
"""

from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torchvision.models.densenet import DenseNet
from torchvision.models.resnet import ResNet


class HamNet(torch.nn.Module, ABC):
    """Abstract network that fuses CNN image features with tabular metadata.

    The network expects a vision backbone that outputs a fixed-length feature
    vector per image. Metadata (sex, age, site) is processed with a small MLP
    and concatenated with image features before classification.
    """

    def __init__(self, backbone: torch.nn.Module, fc_feats: int) -> None:
        """Initialize the model with a backbone and build heads.

        Args:
            backbone: A CNN producing a 1D feature tensor per image. The final
                classifier layer should be replaceable to expose features.
        """
        super().__init__()
        self.fc_feats = fc_feats
        self.backbone, num_features = self.set_backbone(backbone)

        # Small MLP to process 3 metadata features: sex, age, anatom_site
        # Encode 3 metadata scalars into a compact representation
        self.meta_net = torch.nn.Sequential(
            torch.nn.Linear(3, 64),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 128),
            torch.nn.SiLU(),
        )

        # Final classifier over concatenated [image_features || meta_features]
        # Output has 7 logits corresponding to 7 diagnosis classes
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(num_features + 128, self.fc_feats),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(self.fc_feats, 7),
        )

    @abstractmethod
    def set_backbone(self, backbone: torch.nn.Module) -> Tuple[torch.nn.Module, int]:
        """Prepare the backbone for feature extraction.

        Implementations must remove the final classification layer and return
        the modified backbone and the number of output features.

        Args:
            backbone: The torchvision backbone instance.

        Returns:
            A tuple of (prepared_backbone, num_output_features).
        """
        pass

    def forward(self, img: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            img: Tensor of shape (B, C, H, W).
            meta: Tensor of normalized metadata of shape (B, 3).

        Returns:
            Logits tensor of shape (B, 7).
        """
        # Backbone returns pooled per-image feature vectors
        img_features = self.backbone(img)
        meta_features = self.meta_net(meta)
        # Concatenate along feature dimension (B, F_img + F_meta)
        x = torch.cat([img_features, meta_features], dim=1)
        return self.classifier(x)


class HamDenseNet(HamNet):
    """HamNet specialization that uses a torchvision DenseNet backbone."""

    def __init__(self, densenet: DenseNet) -> None:
        super().__init__(backbone=densenet, fc_feats=512)

    def set_backbone(self, backbone: torch.nn.Module) -> Tuple[torch.nn.Module, int]:
        """Replace DenseNet classifier with identity and return feature size."""
        num_features = backbone.classifier.in_features
        # Remove DenseNet's classification layer to expose penultimate features
        backbone.classifier = torch.nn.Identity()
        return backbone, num_features


class HamResNet(HamNet):
    """HamNet specialization that uses a torchvision ResNet backbone."""

    def __init__(self, resnet: ResNet) -> None:
        super().__init__(backbone=resnet, fc_feats=1024)

    def set_backbone(self, backbone: torch.nn.Module) -> Tuple[torch.nn.Module, int]:
        """Replace ResNet fully-connected head with identity and return feature size."""
        num_features = backbone.fc.in_features
        # Replace the final FC layer with identity to get pooled features
        backbone.fc = torch.nn.Identity()
        return backbone, num_features
