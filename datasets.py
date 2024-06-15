from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import torch
import os
import config

__all__ = [
    "create_train_loader",
    "create_test_loader",
    "create_validation_loader",
]

train_transform = v2.Compose([
    v2.Resize((config.IMAGE_WIDTH, config.IMAGE_HEIGHT)),
    v2.RandomRotation(degrees=(-25, 30)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=False),
    v2.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])
])

test_transform = v2.Compose([
    v2.Resize((config.IMAGE_WIDTH, config.IMAGE_HEIGHT)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=False),
    v2.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])
])

def create_train_loader():
    ds = ImageFolder(os.path.join(config.DATASET_PATH, "Train"), transform=train_transform)
    return DataLoader(ds, batch_size=config.TRAINING_BATCH_SIZE, shuffle=True, drop_last=True)

def create_test_loader():
    ds = ImageFolder(os.path.join(config.DATASET_PATH, "Test"), transform=test_transform)
    return DataLoader(ds, batch_size=config.TEST_BATCH_SIZE, shuffle=True, drop_last=True)

def create_validation_loader():
    ds = ImageFolder(os.path.join(config.DATASET_PATH, "Validation"), transform=test_transform)
    return DataLoader(ds, batch_size=config.VALIDATION_BATCH_SIZE, shuffle=True, drop_last=False)
