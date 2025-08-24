"""Data structures and preprocessing utilities for HAM10000 dataset.

Provides `HamImage` records, metadata loading/cleaning, normalization helpers,
and a `Dataset` for classification with optional augmentations.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from hamnet.constants import ANATOM_SITE_MAPPING, DIAGNOSIS_MAPPING, SEX_MAPPING


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
    def __init__(self, images: List[HamImage], train: bool) -> None:
        self.images = images
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        ages = [img.age for img in images]
        self.mean_age, self.std_age = compute_mean_std(ages)

        sexes = [SEX_MAPPING[img.sex] for img in images]
        self.mean_sex, self.std_sex = compute_mean_std(sexes)

        sites = [ANATOM_SITE_MAPPING[img.anatom_site] for img in images]
        self.mean_site, self.std_site = compute_mean_std(sites)

        if not train:
            self.transforms = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((500, 500)),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]
            )
        else:
            self.transforms = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((500, 500)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]
            )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        image = self.images[index]
        image_path = image.path / f"{image.identifier}.jpg"
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transforms(img)

        sex = SEX_MAPPING[image.sex]
        diagnosis = DIAGNOSIS_MAPPING[image.diagnosis]
        site = ANATOM_SITE_MAPPING[image.anatom_site]
        age = float(image.age)

        meta = normalize_meta(
            values=[sex, age, site],
            means=[self.mean_sex, self.mean_age, self.mean_site],
            stds=[self.std_sex, self.std_age, self.std_site],
        )
        return img, meta, diagnosis
