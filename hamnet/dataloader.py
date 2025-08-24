from typing import List, Tuple

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from hamnet.constants import SEED
from hamnet.preprocessing import DIAGNOSIS_MAPPING, HamImage, HamImageDiagnosisDataset


def get_train_test_val_split(
    images: List[HamImage],
) -> Tuple[List[HamImage], List[HamImage], List[HamImage]]:
    labels = [DIAGNOSIS_MAPPING[img.diagnosis] for img in images]

    train_images, temp_images, _, temp_labels = train_test_split(
        images, labels, test_size=0.2, stratify=labels, random_state=SEED
    )

    val_images, test_images, _, _ = train_test_split(
        temp_images,
        temp_labels,
        test_size=0.5,
        stratify=temp_labels,
        random_state=SEED,
    )

    return train_images, test_images, val_images


def get_dataloader(
    train_images: List[HamImage],
    test_images: List[HamImage],
    val_images: List[HamImage],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train = HamImageDiagnosisDataset(train_images, train=True)
    test = HamImageDiagnosisDataset(test_images, train=False)
    val = HamImageDiagnosisDataset(val_images, train=False)

    g = torch.Generator()
    g.manual_seed(SEED)

    trainloader = DataLoader(
        train, shuffle=True, batch_size=32, generator=g, num_workers=2
    )
    testloader = DataLoader(test, shuffle=False, batch_size=32, num_workers=2)
    valloader = DataLoader(val, shuffle=False, batch_size=32, num_workers=2)

    return trainloader, testloader, valloader
