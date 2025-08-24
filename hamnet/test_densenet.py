"""Inference script to evaluate a pretrained HamDenseNet checkpoint on test split."""

from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from torchvision import models

from hamnet.constants import SEED
from hamnet.dataloader import get_dataloader, get_train_test_val_split
from hamnet.hamnet import HamDenseNet
from hamnet.preprocessing import concat_metadata, load_metadata
from hamnet.utils import safe_load_into_ham, seed_everything, test_evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    seed_everything(SEED)

    densenet = models.densenet121(weights=None)
    model = HamDenseNet(densenet=densenet)
    model.to(device)

    pth_path = hf_hub_download("galactixx/Ham-DenseNet", "ham-densenet.bin")
    # Load checkpoint and fix prefix to match current backbone naming
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

    acc = test_evaluate(model=model, loader=testloader, device=device)
    print(f"Accuracy: {acc:.3f}..")
