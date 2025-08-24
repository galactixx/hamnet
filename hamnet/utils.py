from pathlib import Path

import torch

from hamnet.hamnet import HamNet


def safe_load_into_ham(
    model: HamNet, path: Path, device: torch.device, layer_prefix: str
) -> HamNet:
    sd = torch.load(path, map_location=torch.device(device), weights_only=True)

    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    new_sd = {}
    for k, v in sd.items():
        nk = k

        # Common prefix fixes
        if nk.startswith(layer_prefix):
            nk = "backbone." + nk[len(layer_prefix) :]

        new_sd[nk] = v

    model.load_state_dict(new_sd, strict=True)
    return model
