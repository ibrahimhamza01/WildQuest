"""Manifest-backed image loading; test splits are intentionally inaccessible."""

import csv
import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ALLOWED_SPLITS = frozenset(("train", "val"))


def image_transforms(split):
    """Light augmentation for training; pretrained-weight preprocessing for validation."""
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.75, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    if split == "val":
        return EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()
    raise ValueError("Training may only read the 'train' or 'val' split")


class WildlifeDataset(Dataset):
    """Read exact class IDs and local image paths from the frozen manifest."""

    def __init__(self, repo_root, split, transform=None, manifest_path=None, mapping_path=None):
        if split not in ALLOWED_SPLITS:
            raise ValueError("Only train/val are permitted; test sets are held out")
        root = Path(repo_root).resolve()
        manifest = Path(manifest_path) if manifest_path else root / "Code/configs/dataset_manifest.csv"
        mapping = Path(mapping_path) if mapping_path else root / "Code/configs/class_mapping.json"
        self.repo_root = root
        self.split = split
        self.transform = transform if transform is not None else image_transforms(split)

        labels = json.loads(mapping.read_text(encoding="utf-8"))["id_to_label"]
        if set(labels) != {str(i) for i in range(25)}:
            raise ValueError("Expected the existing 25-class mapping")
        self.id_to_label = labels
        self.samples = []
        with manifest.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            required = {"class_id", "split", "local_path", "scientific_name"}
            if not required.issubset(reader.fieldnames or ()):
                raise ValueError("Manifest lacks required columns")
            for row in reader:
                if row["split"] != split:
                    continue
                class_id = int(row["class_id"])
                if class_id not in range(25) or row["scientific_name"] != labels[str(class_id)]:
                    raise ValueError("Manifest/class mapping mismatch")
                relative = Path(row["local_path"])
                if (relative.is_absolute() or ".." in relative.parts
                        or relative.parent != Path("data/raw/images") / split / f"{class_id:02d}"):
                    raise ValueError(f"Unsafe or unexpected image path: {relative}")
                self.samples.append((root / relative, class_id))
        if not self.samples:
            raise ValueError(f"No {split} rows found in {manifest}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, label


def make_dataloaders(repo_root, batch_size=32, num_workers=4, seed=42):
    """Construct *only* training and validation loaders (never held-out tests)."""
    import torch
    train = WildlifeDataset(repo_root, "train")
    val = WildlifeDataset(repo_root, "val")
    use_cuda = torch.cuda.is_available()
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=use_cuda, generator=generator,
    )
    val_loader = DataLoader(
        val, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=use_cuda, generator=generator,
    )
    return train_loader, val_loader
