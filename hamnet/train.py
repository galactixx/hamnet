import os
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage
from torchvision import models
from torchvision.models.densenet import DenseNet
from tqdm import tqdm

from hamnet.constants import SEED
from hamnet.dataloader import get_dataloader, get_train_test_val_split
from hamnet.hamnet import HamDenseNet
from hamnet.preprocessing import DIAGNOSIS_MAPPING, concat_metadata, load_metadata

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def unfreeze_layer(model: DenseNet, layer: str) -> None:
    for name, param in model.named_parameters():
        if name.startswith(layer):
            param.requires_grad = True

    for name, module in model.named_modules():
        if name.startswith(layer) and isinstance(module, torch.nn.BatchNorm2d):
            module.train()


@dataclass(frozen=True)
class ParamGroup:
    layer: str
    epoch: int
    params: torch.nn.Parameter
    lr: float
    momentum: float
    decay: float

    @property
    def group(self) -> Dict[str, Any]:
        return {
            "params": self.params,
            "lr": self.lr,
            "momentum": self.momentum,
            "weight_decay": self.decay,
        }


class ProgressiveUnfreezer:
    def __init__(
        self, model: DenseNet, optimizer: SGD, params: List[ParamGroup]
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.params = deque(sorted(params, key=lambda x: x.epoch))

    def unfreeze(self, epoch: int) -> None:
        if not len(self.params):
            return None

        top = self.params[0]
        if epoch == top.epoch:
            unfreeze_layer(model=self.model, layer=top.layer)
            self.optimizer.add_param_group(top.group)
            self.params.popleft()
            print("Unfreezing layer...")


def train_evaluate(
    model: DenseNet,
    loader: DataLoader,
    criterion: CrossEntropyLoss,
    ema: ExponentialMovingAverage,
) -> Tuple[int, int]:
    model.eval()
    ema.store()
    ema.copy_to()
    correct = total = 0
    running_loss = 0.0

    with torch.no_grad():
        for imgs, meta, labels in tqdm(loader, desc=f"Evaluation: "):
            imgs, meta, labels = imgs.to(device), meta.to(device), labels.to(device)

            logits = model(imgs, meta)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            running_loss += loss.item() * imgs.size(0)

    ema.restore()
    eval_loss = running_loss / len(loader.dataset)
    return eval_loss, correct / total


if __name__ == "__main__":
    os.environ["PYTHONHASHSEED"] = str(SEED)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    ham_dir = Path("ham")
    metadata = concat_metadata(paths=[ham_dir])
    images = load_metadata(metadata=metadata)

    train_images, test_images, val_images = get_train_test_val_split(images=images)

    trainloader, testloader, valloader = get_dataloader(
        train_images=train_images,
        test_images=test_images,
        val_images=val_images,
    )

    pth_path = hf_hub_download("galactixx/Ham-DenseNet", "densenet121-a639ec97.bin")
    densenet = models.densenet121(weights=None)
    densenet.load_state_dict(torch.load(pth_path))

    for name, param in densenet.named_parameters():
        param.requires_grad = False

    for name, module in densenet.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()

    model = HamDenseNet(densenet=densenet)
    model.to(device)

    train_labels = [DIAGNOSIS_MAPPING[img.diagnosis] for img in train_images]
    num_classes = len(list(set(train_labels)))
    counts = torch.bincount(torch.tensor(train_labels), minlength=num_classes)
    weights = counts.sum() / (num_classes * counts.clamp_min(1))

    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)

    EPOCHS = 100

    unfreeze_layer(model=model.densenet.features, layer="classifier")
    optimizer = SGD(
        [
            {
                "params": model.classifier.parameters(),
                "lr": 1e-3,
                "momentum": 0.9,
                "weight_decay": 1e-4,
            },
            {
                "params": model.meta_net.parameters(),
                "lr": 1e-3,
                "momentum": 0.9,
                "weight_decay": 1e-4,
            },
        ]
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=3)
    criterion = CrossEntropyLoss(
        label_smoothing=0.05, weight=torch.tensor(weights).to(device)
    )

    patience = 7
    best_loss = float("inf")
    no_improve = 0

    scaler = GradScaler()
    unfreezer = ProgressiveUnfreezer(
        model.densenet.features,
        optimizer,
        [
            ParamGroup(
                layer="denseblock4",
                epoch=3,
                params=model.densenet.features.denseblock4.parameters(),
                lr=5e-3,
                momentum=0.9,
                decay=1e-5,
            ),
            ParamGroup(
                layer="denseblock3",
                epoch=6,
                params=model.densenet.features.denseblock3.parameters(),
                lr=3e-3,
                momentum=0.9,
                decay=1e-5,
            ),
            ParamGroup(
                layer="denseblock2",
                epoch=9,
                params=model.densenet.features.denseblock2.parameters(),
                lr=1e-3,
                momentum=0.9,
                decay=1e-5,
            ),
        ],
    )

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for imgs, meta, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs, meta, labels = imgs.to(device), meta.to(device), labels.to(device)

            optimizer.zero_grad()

            with autocast():
                preds = model(imgs, meta)
                loss = criterion(preds, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ema.update()

            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / len(trainloader.dataset)

        val_loss, val_acc = train_evaluate(model, valloader, criterion, ema)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{EPOCHS}.. "
            f"Train loss: {train_loss:.3f}.. "
            f"Val loss: {val_loss:.3f}.. "
            f"Accuracy: {val_acc:.3f}.."
        )

        if val_loss < best_loss:
            no_improve = 0
            best_loss = val_loss
        else:
            no_improve += 1
            if no_improve >= patience:
                break

        unfreezer.unfreeze(epoch=epoch + 1)
        torch.cuda.empty_cache()
