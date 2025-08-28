"""Utility helpers for training and evaluation.

Contains progressive unfreezing utilities, checkpoint-safe load helpers,
evaluation, and seeding for reproducibility.
"""

import os
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torchvision.transforms.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from hamnet.hamnet import HamNet


def unfreeze_layer(model: torch.nn.Module, layer: str) -> None:
    """Unfreeze parameters and enable training for BatchNorm in a layer prefix."""
    for name, param in model.named_parameters():
        if name.startswith(layer):
            # Allow gradients to flow for these parameters
            param.requires_grad = True

    for name, module in model.named_modules():
        if name.startswith(layer) and isinstance(module, torch.nn.BatchNorm2d):
            # BatchNorm needs train mode to update running stats after unfreezing
            module.train()


@dataclass(frozen=True)
class ParamGroup:
    """Optimizer hyperparameters associated with a layer to unfreeze at an epoch."""

    layer: str
    epoch: int
    params: torch.nn.Parameter
    lr: float
    decay: float

    @property
    def group(self) -> Dict[str, Any]:
        return {
            "params": self.params,
            "lr": self.lr,
            "weight_decay": self.decay,
        }


class ProgressiveUnfreezer:
    """Manage progressive unfreezing by epoch and add optimizer param groups."""

    def __init__(self, model: torch.nn.Module, params: List[ParamGroup]) -> None:
        self.model = model
        self.params = deque(sorted(params, key=lambda x: x.epoch))
        self._optimizer: Optional[AdamW] = None

    @property
    def optimizer(self) -> AdamW:
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: AdamW) -> None:
        self._optimizer = optimizer

    def unfreeze(self, epoch: int) -> None:
        """If the current epoch matches the schedule, unfreeze next layer."""
        # Nothing left to unfreeze
        if not len(self.params):
            return None

        top = self.params[0]
        if epoch == top.epoch:
            unfreeze_layer(model=self.model, layer=top.layer)
            # Start optimizing the newly unfrozen parameters
            self.optimizer.add_param_group(top.group)
            self.params.popleft()
            print("Unfreezing layer...")


def safe_load_into_ham(
    model: HamNet, path: Path, device: torch.device, layer_prefix: str
) -> HamNet:
    """Load a checkpoint into `HamNet` fixing layer prefixes for the backbone."""
    # Load checkpoint dictionary or plain state_dict
    sd = torch.load(path, map_location=torch.device(device), weights_only=True)

    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    new_sd = {}
    for k, v in sd.items():
        nk = k

        # Common prefix fixes to align checkpoint keys with current model
        if nk.startswith(layer_prefix):
            nk = "backbone." + nk[len(layer_prefix) :]

        new_sd[nk] = v

    # Strict=True ensures missing/mismatched keys fail fast
    model.load_state_dict(new_sd, strict=True)
    return model


def test_evaluate(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> float:
    """Compute accuracy over a dataloader without gradient calculations."""
    tta_transforms = [
        lambda x: x,
        lambda x: F.hflip(x),
        lambda x: F.vflip(x),
        lambda x: F.rotate(x, 15),
        lambda x: F.adjust_brightness(x, 0.2),
        lambda x: F.adjust_contrast(x, 0.5),
        lambda x: F.rotate(x, -15),
    ]

    correct = total = 0

    with torch.no_grad():
        for imgs, meta, labels in tqdm(loader, desc=f"Evaluation: "):
            base_imgs, meta, labels = (
                imgs.to(device),
                meta.to(device),
                labels.to(device),
            )

            probs = []

            for aug in tta_transforms:
                imgs = torch.stack([aug(img.cpu()) for img in base_imgs]).to(device)
                logits = model(imgs, meta)
                # Collect softmax probabilities for test-time augmentation
                probs.append(torch.softmax(logits, dim=1))

            # Average probabilities across TTA variants and pick argmax
            preds = torch.stack(probs).mean(0).argmax(1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for reproducibility (incl. CUDA)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Make CUDA algorithms deterministic where possible
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
