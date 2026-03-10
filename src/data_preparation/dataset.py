from pathlib import Path

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

DATA_SPLITS_DIR = Path("data/splits")

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0  # safe default for MacBook


def get_transforms(image_size: int = IMAGE_SIZE):
    base_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return {
        "train": base_transform,
        "val": base_transform,
        "test": base_transform,
    }


def get_datasets(image_size: int = IMAGE_SIZE):
    transform_dict = get_transforms(image_size)

    train_dataset = datasets.ImageFolder(
        root=DATA_SPLITS_DIR / "train",
        transform=transform_dict["train"]
    )

    val_dataset = datasets.ImageFolder(
        root=DATA_SPLITS_DIR / "val",
        transform=transform_dict["val"]
    )

    test_dataset = datasets.ImageFolder(
        root=DATA_SPLITS_DIR / "test",
        transform=transform_dict["test"]
    )

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(
    image_size: int = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS
):
    train_dataset, val_dataset, test_dataset = get_datasets(image_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, train_dataset.classes