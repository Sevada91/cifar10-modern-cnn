from torchvision import datasets
from torch.utils.data import DataLoader, random_split

from .transforms import get_transforms
from .dataset import TransformDataset


def create_dataloaders(
    data_root: str = "../../data",
    batch_size: int = 64,
    num_workers: int = 4,
    val_ratio: float = 0.1,
    augment: bool = True,
    return_datasets: bool = False
):
    """
    Create and return PyTorch DataLoaders for CIFAR-10.

    This function:
    - loads the CIFAR-10 train/test datasets from disk
    - splits the train set into train/validation subsets
    - applies augmentation only to the training subset
    - builds DataLoaders for train, validation, and test

    Args:
        data_root (str): Path to the CIFAR-10 data directory.
        batch_size (int): Batch size for all DataLoaders.
        num_workers (int): Number of workers for DataLoaders.
        val_ratio (float): Fraction of the train set to use as validation.
        augment (bool): Whether to apply data augmentation to the train set.

    Returns:
        train_loader (DataLoader): DataLoader for the training split.
        val_loader (DataLoader): DataLoader for the validation split.
        test_loader (DataLoader): DataLoader for the CIFAR-10 test set.
    """

    # Get transforms: train has augmentation (if augment=True), test has no augmentation
    train_transform, test_transform = get_transforms(augment=augment)

    # Base train dataset without transform (we'll apply transforms via wrapper)
    base_train = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=False,
        transform=None,
    )

    # Compute train/val split sizes
    n_total = len(base_train)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val

    train_base, val_base = random_split(base_train, [n_train, n_val])

    # Wrap with different transforms:
    # - train: augmented transform
    # - val: test-style transform (no random augmentation)
    train_dataset = TransformDataset(train_base, train_transform)
    val_dataset = TransformDataset(val_base, test_transform)

    # Test dataset uses the non-augmented transform
    test_dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=False,
        transform=test_transform,
    )

    # Build DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    if return_datasets:
        return train_dataset, val_dataset, test_dataset
    else:
        return train_loader, val_loader, test_loader
