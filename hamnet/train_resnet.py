import torch
from huggingface_hub import hf_hub_download
from torchvision import models

from hamnet.hamnet import HamResNet
from hamnet.train import train
from hamnet.utils import ParamGroup, ProgressiveUnfreezer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    pth_path = hf_hub_download(
        "galactixx/Torchvision-ResNet50-11ad3fa6", "resnet50-11ad3fa6.bin"
    )
    resnet = models.resnet50(weights=None)
    resnet.load_state_dict(torch.load(pth_path, map_location=torch.device("cpu")))

    for name, param in resnet.named_parameters():
        param.requires_grad = False

    for name, module in resnet.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()

    model = HamResNet(resnet=resnet)

    unfreezer = ProgressiveUnfreezer(
        model.backbone,
        [
            ParamGroup(
                layer="layer4",
                epoch=3,
                params=model.backbone.layer4.parameters(),
                lr=5e-3,
                momentum=0.9,
                decay=1e-5,
            ),
            ParamGroup(
                layer="layer3",
                epoch=6,
                params=model.backbone.layer3.parameters(),
                lr=3e-3,
                momentum=0.9,
                decay=1e-5,
            ),
            ParamGroup(
                layer="layer2",
                epoch=9,
                params=model.backbone.layer2.parameters(),
                lr=1e-3,
                momentum=0.9,
                decay=1e-5,
            ),
        ],
    )
    train(model=model, unfreezer=unfreezer)
