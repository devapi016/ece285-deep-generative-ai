"""Loads the Kaggle chest X-ray images into memory as a cached uint8 tensor.

The full-resolution JPEGs are large (often >1500 px), so decoding them every epoch
would dominate training time. They are decoded once, resized, and cached to disk.
"""

from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from config import (
    BATCH_SIZE,
    CACHE_DIR,
    CLASS_NAMES,
    DATA_ROOT,
    HELDOUT_SPLIT,
    IMAGE_SIZE,
    NUM_DECODE_WORKERS,
    TRAIN_SPLITS,
    TRAINING_CLASS,
)

IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}
PIXEL_MAX = 255.0


def _list_split_images(split: str) -> list[tuple[Path, int]]:
    return [
        (path, label)
        for label, class_name in enumerate(CLASS_NAMES)
        for path in sorted((DATA_ROOT / split / class_name).iterdir())
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def _decode_image(path: Path) -> np.ndarray:
    # Resize the whole film (no crop) so lung apices and bases are never cut off;
    # the mild aspect-ratio change is consistent across real and generated images.
    with Image.open(path) as image:
        resized = image.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
        return np.asarray(resized, dtype=np.uint8)


def _load_split(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    entries = _list_split_images(split)
    with Pool(NUM_DECODE_WORKERS) as pool:
        pixels = pool.map(_decode_image, [path for path, _ in entries], chunksize=32)
    images = torch.from_numpy(np.stack(pixels)).unsqueeze(1)  # (N, 1, H, W)
    labels = torch.tensor([label for _, label in entries])
    return images, labels


def load_split_uint8(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (images uint8 [N,1,H,W], labels [N]) for one dataset split, using the cache."""
    cache_path = CACHE_DIR / f"{split}_{IMAGE_SIZE}px.pt"
    if cache_path.exists():
        cached = torch.load(cache_path)
        return cached["images"], cached["labels"]

    images, labels = _load_split(split)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"images": images, "labels": labels}, cache_path)
    return images, labels


def load_images(splits: tuple[str, ...]) -> torch.Tensor:
    """Concatenates the TRAINING_CLASS images of the given splits as floats in [0, 1]."""
    class_label = CLASS_NAMES.index(TRAINING_CLASS)
    class_images = []
    for split in splits:
        images, labels = load_split_uint8(split)
        class_images.append(images[labels == class_label])
    return torch.cat(class_images).float() / PIXEL_MAX


def make_loaders() -> tuple[DataLoader, DataLoader]:
    """Builds the training loader (train + val splits) and the held-out test loader."""
    train_images = load_images(TRAIN_SPLITS)
    heldout_images = load_images((HELDOUT_SPLIT,))
    train_loader = DataLoader(
        TensorDataset(train_images), batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    heldout_loader = DataLoader(TensorDataset(heldout_images), batch_size=BATCH_SIZE)
    return train_loader, heldout_loader
