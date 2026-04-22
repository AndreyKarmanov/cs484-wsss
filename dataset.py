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


def _make_voc_palette(N: int = 256) -> np.ndarray:
    """Standard PASCAL VOC 256-color palette as an [N, 3] uint8 array.

    Generated algorithmically with the same bit-interleaving recipe used by the
    VOC devkit (see e.g. `VOClabelcolormap.m`). Index 0 is background, indices
    1..20 are the foreground classes (in the canonical VOC order matching
    `VOC_CLASSES`), and index 255 is the void/ignore color (white).
    """
    palette = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        palette[i] = (r, g, b)
    return palette


VOC_PALETTE = _make_voc_palette()


def colorize(label_map: np.ndarray) -> np.ndarray:
    """Map a [H, W] integer label map to an [H, W, 3] uint8 RGB image.

    Indices 0..20 are background + 20 VOC classes; 255 (void) renders as white.
    """
    label_map = np.clip(label_map, 0, len(VOC_PALETTE) - 1)
    return VOC_PALETTE[label_map]


def make_transform(resize_size: int | tuple[int, int] = 448):
    size = (resize_size, resize_size) if isinstance(resize_size, int) else resize_size
    return v2.Compose([
        v2.ToImage(),
        v2.Resize(size, antialias=True),
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


def make_voc_datasets(root: str = "./data", resize_size: int | tuple[int, int] = 448):
    transform = make_transform(resize_size)
    voc_train = torchvision.datasets.VOCSegmentation(
        root=root, year="2012", image_set="train", download=False
    )
    voc_val = torchvision.datasets.VOCSegmentation(
        root=root, year="2012", image_set="val", download=False
    )
    return WSSSDataset(voc_train, transform), WSSSDataset(voc_val, transform)
