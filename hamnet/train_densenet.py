"""Entry point script to train HamDenseNet with progressive unfreezing."""

import shutil
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from torchvision import models
from torchvision.models import DenseNet121_Weights
from torchvision.models.densenet import DenseNet

from hamnet.hamnet import HamFiLMDenseNet
from hamnet.train import train
from hamnet.utils import ParamGroup, ProgressiveUnfreezer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_densenet() -> DenseNet:
    src = Path(
        hf_hub_download(
            "galactixx/Torchvision-DenseNet121-a639ec97 ", "densenet121-a639ec97.bin"
        )
    )

    # Torch cache path for torchvision checkpoints
    cache = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
    cache.mkdir(parents=True, exist_ok=True)

    # Copy the weights into cache if not already present
    dst = cache / src.name
    if not dst.exists():
        shutil.copy2(src, dst)

    densenet = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
    densenet.load_state_dict(torch.load(dst, map_location=torch.device(device)))
    return densenet


if __name__ == "__main__":
    densenet = get_densenet()
    model = HamFiLMDenseNet(densenet=densenet)

    # Freeze backbone parameters before progressive unfreezing schedule
    for _, param in model.backbone.named_parameters():
        param.requires_grad = False

    # Keep BatchNorm layers in eval mode until their blocks are unfrozen
    for _, module in model.backbone.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()

    # Define which DenseNet blocks to unfreeze and when during training
    unfreezer = ProgressiveUnfreezer(
        model.backbone.features,
        [
            ParamGroup(
                layer="denseblock4",
                epoch=3,
                params=model.backbone.features.denseblock4.parameters(),
                lr=1e-4,
                decay=1e-5,
            ),
            ParamGroup(
                layer="denseblock3",
                epoch=6,
                params=model.backbone.features.denseblock3.parameters(),
                lr=1e-4,
                decay=1e-5,
            ),
            ParamGroup(
                layer="denseblock2",
                epoch=9,
                params=model.backbone.features.denseblock2.parameters(),
                lr=1e-4,
                decay=1e-5,
            ),
        ],
    )
    train(model=model, unfreezer=unfreezer)
