from torch.utils.data import Dataset

class TransformDataset(Dataset):
    """
    Wrap a base dataset (or Subset) and apply a transform on the fly.

    This allows us to:
    - split the base train dataset into train/val
    - apply different transforms to each split (e.g., augment train, not val)
    """

    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label
