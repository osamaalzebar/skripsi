#!/usr/bin/env python3
# dataset_inceptionV3.py

import csv
import os
import random
from typing import Callable, List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class RandomRotate90:
    """Rotate by 0°, 90°, 180°, or 270° randomly."""
    def __call__(self, img):
        k = random.randint(0, 3)
        return img.rotate(90 * k)


def build_transforms(img_size: int = 299, train: bool = True) -> Callable:
    # Inception-v3 expects 299x299
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            RandomRotate90(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


class BrainCSVSet(Dataset):
    """
    CSV format:
      header: image_name,label
      rows:   <filename>,<int_label in {0,1,2,3}>

    image_root points to the folder containing the image files.
    """

    def __init__(self, image_root: str, csv_path: str, train: bool, img_size: int = 299):
        self.image_root = str(image_root)
        self.csv_path = str(csv_path)
        self.items: List[Tuple[str, int]] = []

        with open(self.csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)

            # Normalize headers (strip whitespace)
            field_map = {name.strip(): name for name in reader.fieldnames}
            img_key = field_map.get("image_name")
            lbl_key = field_map.get("label")

            if img_key is None or lbl_key is None:
                raise ValueError("CSV must have headers 'image_name' and 'label'.")

            for row in reader:
                img_name = row[img_key].strip()
                label = int(row[lbl_key])

                if label not in {0, 1, 2, 3}:
                    raise ValueError(f"Unexpected label {label} for row {row}")

                img_path = os.path.join(self.image_root, img_name)
                self.items.append((img_path, label))

        self.tf = build_transforms(img_size=img_size, train=train)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        img = self.tf(img)
        return img, torch.tensor(label, dtype=torch.long), path
