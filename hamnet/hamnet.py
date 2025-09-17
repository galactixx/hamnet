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


class HamFiLMNet(torch.nn.Module, ABC):
    """Abstract network that fuses CNN image features with tabular metadata.

    The network expects a vision backbone that outputs a fixed-length feature
    vector per image. Metadata (sex, age, site) is processed with a small MLP
    and concatenated with image features before classification.
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        fc_feats: int,
        num_classes: int = 7,
        meta_hidden: int = 64,
        keep_meta_concat: bool = True,
    ) -> None:
        super().__init__()
        self.backbone, d = self.set_backbone(backbone)

        self.site_embeds = torch.nn.Embedding(num_embeddings=7, embedding_dim=4)

        # Small MLP to process 3 metadata features: sex, age, anatom_site
        self.meta_embed = torch.nn.Sequential(
            torch.nn.Linear(6, 64), torch.nn.SiLU(), torch.nn.Linear(64, 128)
        )

        # FiLM gate via MLP
        self.gate_mlp = torch.nn.Sequential(
            torch.nn.Linear(6, meta_hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(meta_hidden, d),
        )

        # Concatenate meta features if needed
        self.keep_meta_concat = keep_meta_concat

        # Final classifier over concatenated [image_features || meta_features]
        in_dim = d + (128 if keep_meta_concat else 0)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(in_dim, fc_feats),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(fc_feats, num_classes),
        )

    @abstractmethod
    def set_backbone(self, backbone: torch.nn.Module) -> Tuple[torch.nn.Module, int]:
        """Prepare the backbone for feature extraction.

        Implementations must remove the final classification layer and return
        the modified backbone and the number of output features.
        """
        pass

    def forward(self, img: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        """Forward pass via a gated FiLM layer and a concatenated small MLP."""
        x = self.backbone(img)
        site_embeds = self.site_embeds(site.long())
        meta_new = torch.cat([meta, site_embeds], dim=1)

        gamma = torch.sigmoid(self.gate_mlp(meta_new))
        x = x * gamma

        if self.keep_meta_concat:
            m = self.meta_embed(meta_new)
            x = torch.cat([x, m], dim=1)

        return self.head(x)


class HamFiLMDenseNet(HamFiLMNet):
    """HamNet specialization that uses a torchvision DenseNet backbone."""

    def __init__(self, densenet: DenseNet) -> None:
        super().__init__(backbone=densenet, fc_feats=512)

    def set_backbone(self, backbone: torch.nn.Module) -> Tuple[torch.nn.Module, int]:
        """Replace DenseNet classifier with identity and return feature size."""
        num_features = backbone.classifier.in_features
        backbone.classifier = torch.nn.Identity()
        return backbone, num_features


class HamFiLMResNet(HamFiLMNet):
    """HamNet specialization that uses a torchvision ResNet backbone."""

    def __init__(self, resnet: ResNet) -> None:
        super().__init__(backbone=resnet, fc_feats=1024)

    def set_backbone(self, backbone: torch.nn.Module) -> Tuple[torch.nn.Module, int]:
        """Replace ResNet fully-connected head with identity and return feature size."""
        num_features = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()
        return backbone, num_features
