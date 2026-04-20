import numpy as np
import torch
import torchvision
from torchvision.transforms import v2


VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
    'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
    'sofa', 'train', 'tvmonitor'
]


def make_transform(resize_size: int = 448):
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((resize_size, resize_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


class WSSSDataset(torch.utils.data.Dataset):
    """Wraps VOCSegmentation to also return the dataset index and an image-level
    label vector, so DINO features can be cached by index during training."""

    def __init__(self, voc_dataset, transform):
        self.voc = voc_dataset
        self.transform = transform

    def __len__(self):
        return len(self.voc)

    def __getitem__(self, idx):
        img, mask = self.voc[idx]
        img_t = self.transform(img)

        mask_t = torch.tensor(np.array(mask), dtype=torch.long)
        unique_classes = torch.unique(mask_t)
        unique_classes = unique_classes[(unique_classes != 0) & (unique_classes != 255)]

        label_vec = torch.zeros(20, dtype=torch.float32)
        if len(unique_classes) > 0:
            label_vec[unique_classes - 1] = 1.0

        return idx, img_t, label_vec


def wsss_collate_fn(batch):
    indices = torch.tensor([b[0] for b in batch], dtype=torch.long)
    images  = torch.stack([b[1] for b in batch])
    labels  = torch.stack([b[2] for b in batch])
    return indices, images, labels


def make_voc_datasets(root: str = "./data"):
    transform = make_transform()
    voc_train = torchvision.datasets.VOCSegmentation(
        root=root, year="2012", image_set="train", download=False
    )
    voc_val = torchvision.datasets.VOCSegmentation(
        root=root, year="2012", image_set="val", download=False
    )
    return WSSSDataset(voc_train, transform), WSSSDataset(voc_val, transform)
