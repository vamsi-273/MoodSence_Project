import os
from torch.utils.data import DataLoader
from torchvision import datasets
from src.preprocessing.transforms import train_transforms, val_transforms


def get_dataloaders(batch_size=16):
    base_path = "data/raw/raf_db"

    train_dataset = datasets.ImageFolder(
        root=os.path.join(base_path, "train"),
        transform=train_transforms
    )

    val_dataset = datasets.ImageFolder(
        root=os.path.join(base_path, "test"),
        transform=val_transforms
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader