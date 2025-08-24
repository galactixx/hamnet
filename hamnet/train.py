from pathlib import Path
from typing import Tuple

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm

from hamnet.constants import DIAGNOSIS_MAPPING, SEED
from hamnet.dataloader import get_dataloader, get_train_test_val_split
from hamnet.hamnet import HamNet
from hamnet.preprocessing import concat_metadata, load_metadata
from hamnet.utils import ProgressiveUnfreezer, seed_everything

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_evaluate(
    model: HamNet,
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


def train(model: HamNet, unfreezer: ProgressiveUnfreezer) -> None:
    seed_everything(SEED)

    ham_dir = Path("data/HAM10000")
    metadata = concat_metadata(paths=[ham_dir])
    images = load_metadata(metadata=metadata)

    train_images, test_images, val_images = get_train_test_val_split(images=images)

    trainloader, _, valloader = get_dataloader(
        train_images=train_images,
        test_images=test_images,
        val_images=val_images,
    )

    model.to(device)
    train_labels = [DIAGNOSIS_MAPPING[img.diagnosis] for img in train_images]
    num_classes = len(list(set(train_labels)))
    counts = torch.bincount(torch.tensor(train_labels), minlength=num_classes)
    weights = counts.sum() / (num_classes * counts.clamp_min(1))

    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)

    EPOCHS = 100

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
    unfreezer.optimizer = optimizer
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=3)
    criterion = CrossEntropyLoss(
        label_smoothing=0.05, weight=torch.tensor(weights).to(device)
    )

    patience = 7
    best_loss = float("inf")
    no_improve = 0

    scaler = GradScaler()

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
