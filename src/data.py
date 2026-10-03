import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets


def create_train_val_loaders(
    train_path,
    train_transform,
    test_transform,
    batch_size=32
):
    full_train_dataset = datasets.ImageFolder(
        train_path,
        transform=None
    )

    labels = full_train_dataset.targets

    train_indices, val_indices = train_test_split(
        range(len(full_train_dataset)),
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    train_dataset = datasets.ImageFolder(
        train_path,
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        train_path,
        transform=test_transform
    )

    train_dataset = Subset(
        train_dataset,
        train_indices
    )

    val_dataset = Subset(
        val_dataset,
        val_indices
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    return train_loader, val_loader


def create_test_loader(
    test_path,
    test_transform,
    batch_size=32
):
    test_dataset = datasets.ImageFolder(
        test_path,
        transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    return test_dataset, test_loader