import os
import random
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from hamnet.constants import SEED
from hamnet.dataloader import get_dataloader, get_train_test_val_split
from hamnet.hamnet import HamDenseNet
from hamnet.preprocessing import concat_metadata, load_metadata
from hamnet.utils import safe_load_into_ham

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_evaluate(model: torch.nn.Module, loader: DataLoader) -> float:
    correct = total = 0

    with torch.no_grad():
        for imgs, meta, labels in tqdm(loader, desc=f"Evaluation: "):
            imgs, meta, labels = imgs.to(device), meta.to(device), labels.to(device)

            logits = model(imgs, meta)
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


if __name__ == "__main__":
    os.environ["PYTHONHASHSEED"] = str(SEED)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    densenet = models.densenet121(weights=None)
    model = HamDenseNet(densenet=densenet)
    model.to(device)

    pth_path = hf_hub_download("galactixx/Ham-DenseNet", "ham-densenet.bin")
    model = safe_load_into_ham(model, pth_path, device=device, layer_prefix="densenet.")

    model.eval()

    ham_dir = Path("data/HAM10000")
    metadata = concat_metadata(paths=[ham_dir])
    images = load_metadata(metadata=metadata)

    train_images, test_images, val_images = get_train_test_val_split(images=images)

    _, testloader, _ = get_dataloader(
        train_images=train_images,
        test_images=test_images,
        val_images=val_images,
    )

    acc = test_evaluate(model=model, loader=testloader)
    print(f"Accuracy: {acc:.3f}..")
