import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_transforms(image_size=64, augmentation=False):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if augmentation:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])

    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, test_transform


def get_dataloaders(batch_size=32, image_size=128, augmentation=False):
    train_transform, test_transform = get_transforms(
        image_size=image_size,
        augmentation=augmentation
    )

    trainval_train_transform = datasets.OxfordIIITPet(
        root="data",
        split="trainval",
        target_types="category",
        download=True,
        transform=train_transform
    )

    trainval_test_transform = datasets.OxfordIIITPet(
        root="data",
        split="trainval",
        target_types="category",
        download=True,
        transform=test_transform
    )

    test_dataset = datasets.OxfordIIITPet(
        root="data",
        split="test",
        target_types="category",
        download=True,
        transform=test_transform
    )

    dataset_size = len(trainval_train_transform)

    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(dataset_size, generator=generator).tolist()

    train_size = int(0.8 * dataset_size)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(trainval_train_transform, train_indices)
    val_dataset = Subset(trainval_test_transform, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader