"""Data split and DataLoader creation utilities."""

import random
from typing import List, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from hamnet.constants import DIAGNOSIS_MAPPING, SEED
from hamnet.preprocessing import (
    HamImage,
    HamImageDiagnosisDataset,
    TrainStatistics,
    calculate_statistics,
    get_age,
    get_sex,
    get_site,
)


def get_train_test_val_split(
    images: List[HamImage],
) -> Tuple[List[HamImage], List[HamImage], List[HamImage]]:
    """Split images into train/val/test with stratification on labels."""
    # Convert string diagnoses to numeric labels for stratified splitting
    labels = [DIAGNOSIS_MAPPING[img.diagnosis] for img in images]

    # First split: hold out 20% for val+test (stratified)
    train_images, temp_images, _, temp_labels = train_test_split(
        images, labels, test_size=0.2, stratify=labels, random_state=SEED
    )

    # Second split: split val and test evenly (10% each overall)
    val_images, test_images, _, _ = train_test_split(
        temp_images,
        temp_labels,
        test_size=0.5,
        stratify=temp_labels,
        random_state=SEED,
    )

    # Maintain a consistent return order for downstream code
    return train_images, test_images, val_images


def get_dataloader(
    train_images: List[HamImage],
    test_images: List[HamImage],
    val_images: List[HamImage],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch dataloaders with deterministic shuffling for train set."""
    age = calculate_statistics(train_images, get_age)
    sex = calculate_statistics(train_images, get_sex)
    train_stats = TrainStatistics(age=age, sex=sex)

    def worker_init_fn(worker_id: int) -> None:
        np.random.seed(SEED + worker_id)
        random.seed(SEED + worker_id)

    train = HamImageDiagnosisDataset(train_stats, train_images, train=True)
    test = HamImageDiagnosisDataset(train_stats, test_images, train=False)
    val = HamImageDiagnosisDataset(train_stats, val_images, train=False)

    # Use a generator to make shuffling reproducible across runs
    g = torch.Generator()
    g.manual_seed(SEED)

    # Slightly larger batch for efficiency; workers tuned for common CPUs
    trainloader = DataLoader(
        train,
        shuffle=True,
        batch_size=64,
        generator=g,
        num_workers=2,
        worker_init_fn=worker_init_fn,
    )
    # No shuffling for evaluation splits
    testloader = DataLoader(test, shuffle=False, batch_size=32, num_workers=2)
    valloader = DataLoader(val, shuffle=False, batch_size=32, num_workers=2)

    return trainloader, testloader, valloader
