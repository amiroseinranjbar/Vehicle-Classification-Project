import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset,Sampler
from torchvision import datasets
import random
import torch
from sklearn.model_selection import train_test_split


class BalancedBatchSampler(Sampler):

    def __init__(
        self,
        labels,
        batch_size=32
    ):
        self.labels = labels
        self.batch_size = batch_size

        self.classes = sorted(set(labels))
        self.num_classes = len(self.classes)

        if batch_size % self.num_classes != 0:
            raise ValueError(
                "batch_size must be divisible by number of classes"
            )

        self.samples_per_class = (
            batch_size // self.num_classes
        )

        self.class_indices = {}

        for class_id in self.classes:
            self.class_indices[class_id] = [
                i
                for i, label in enumerate(labels)
                if label == class_id
            ]

        self.num_batches = (
            len(labels) // batch_size
        )

    def __iter__(self):

        class_pools = {}

        for class_id in self.classes:

            indices = self.class_indices[class_id].copy()

            random.shuffle(indices)

            class_pools[class_id] = indices

        for _ in range(self.num_batches):

            batch = []

            for class_id in self.classes:

                pool = class_pools[class_id]

                if len(pool) < self.samples_per_class:

                    indices = self.class_indices[class_id].copy()

                    random.shuffle(indices)

                    pool.extend(indices)

                selected = pool[
                    :self.samples_per_class
                ]

                del pool[
                    :self.samples_per_class
                ]

                batch.extend(selected)

            random.shuffle(batch)

            yield batch

    def __len__(self):
        return self.num_batches


def create_train_val_loaders_balanced(
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

    train_labels = [
        labels[i]
        for i in train_indices
    ]

    train_sampler = BalancedBatchSampler(
        labels=train_labels,
        batch_size=batch_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    return train_loader, val_loader

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