"""Data structures and preprocessing utilities for HAM10000 dataset.

Provides `HamImage` records, metadata loading/cleaning, normalization helpers,
and a `Dataset` for classification with optional augmentations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from hamnet.constants import (
    ANATOM_SITE_MAPPING,
    DIAGNOSIS_MAPPING,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SEX_MAPPING,
)


@dataclass(frozen=True)
class Statistics:
    mean: float
    std: float


@dataclass(frozen=True)
class TrainStatistics:
    age: Statistics
    sex: Statistics


def calculate_statistics(
    images: List[HamImage], getter: Callable[[HamImage], float]
) -> Statistics:
    obs = [getter(img) for img in images]
    mean, std = compute_mean_std(obs)
    return Statistics(mean=mean, std=std)


def get_age(image: HamImage) -> float:
    return image.age


def get_sex(image: HamImage) -> float:
    return SEX_MAPPING[image.sex]


@dataclass(frozen=True)
class HamImage:
    """Lightweight record for an image and its associated metadata."""

    path: Path
    identifier: str
    age: str
    sex: str
    diagnosis: str
    anatom_site: str


def concat_metadata(paths: List[Path]) -> pd.DataFrame:
    """Read and concatenate metadata.csv files from base paths with base_path column."""
    data = pd.DataFrame()
    for path in paths:
        metadata = pd.read_csv(path / "metadata.csv")
        metadata["base_path"] = path.as_posix()
        data = pd.concat([data, metadata])
    data = data.drop_duplicates(["isic_id"])
    return data


def load_metadata(metadata: pd.DataFrame) -> List[HamImage]:
    """Filter required fields and convert rows into `HamImage` records."""
    metadata = metadata[pd.notnull(metadata["age_approx"])]
    metadata = metadata[pd.notnull(metadata["sex"])]
    metadata = metadata[pd.notnull(metadata["anatom_site_general"])]
    metadata = metadata[pd.notnull(metadata["diagnosis_3"])]
    images: List[HamImage] = []
    for _, row in metadata.iterrows():
        images.append(
            HamImage(
                path=Path(row["base_path"]),
                identifier=row["isic_id"],
                age=row["age_approx"],
                sex=row["sex"],
                diagnosis=row["diagnosis_3"],
                anatom_site=row["anatom_site_general"],
            )
        )
    return images


def compute_mean_std(values: List[Any]) -> Tuple[float, float]:
    arr = np.array(values, dtype=np.float32)
    return float(arr.mean()), float(arr.std())


def normalize_meta(
    values: Sequence[float], means: Sequence[float], stds: Sequence[float]
) -> torch.Tensor:
    normed = [(v - m) / s for v, m, s in zip(values, means, stds)]
    return torch.tensor(normed, dtype=torch.float32)


class HamImageDiagnosisDataset(Dataset):
    def __init__(
        self, stats: TrainStatistics, images: List[HamImage], train: bool
    ) -> None:
        self.stats = stats
        self.images = images

        if not train:
            # Deterministic transforms for evaluation
            self.transforms = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((512, 512)),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ]
            )
        else:
            # Light augmentation for regularization during training
            self.transforms = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((512, 512)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ]
            )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        image = self.images[index]
        image_path = image.path / f"{image.identifier}.jpg"
        # Load BGR image via OpenCV and convert to RGB tensor
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transforms(img)

        sex = SEX_MAPPING[image.sex]
        diagnosis = DIAGNOSIS_MAPPING[image.diagnosis]
        site = ANATOM_SITE_MAPPING[image.anatom_site]
        age = float(image.age)

        # Normalize tabular metadata to zero mean, unit variance
        meta = normalize_meta(
            values=[sex, age],
            means=[self.stats.sex.mean, self.stats.age.mean],
            stds=[self.stats.sex.std, self.stats.age.std],
        )
        return img, meta, site, diagnosis
