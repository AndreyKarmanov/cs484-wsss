import numpy as np
import torch
import torchvision
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


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


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def make_transform(resize_size: int | tuple[int, int] = 448):
    size = (resize_size, resize_size) if isinstance(resize_size, int) else resize_size
    return v2.Compose([
        v2.ToImage(),
        v2.Resize(size, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def denorm_to_uint8(img_t: torch.Tensor) -> np.ndarray:
    """ImageNet-normalized CHW float tensor -> HWC uint8."""
    x = img_t.detach().cpu().numpy().transpose(1, 2, 0)
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    x = (x * std + mean).clip(0.0, 1.0)
    return (x * 255.0).astype(np.uint8)


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

        return idx, img_t, label_vec, mask_t


def wsss_collate_fn(batch):
    indices = torch.tensor([b[0] for b in batch], dtype=torch.long)
    images  = torch.stack([b[1] for b in batch])
    labels  = torch.stack([b[2] for b in batch])
    masks   = [b[3] for b in batch]
    return indices, images, labels, masks


def make_voc_datasets(root: str = "./data", resize_size: int | tuple[int, int] = 448):
    transform = make_transform(resize_size)
    voc_train = torchvision.datasets.VOCSegmentation(
        root=root, year="2012", image_set="train", download=True
    )
    voc_val = torchvision.datasets.VOCSegmentation(
        root=root, year="2012", image_set="val", download=True
    )
    return WSSSDataset(voc_train, transform), WSSSDataset(voc_val, transform)


def get_wsss_dataloaders(
    train_ds,
    val_ds,
    batch_size: int = 16
):
    """
    Returns train_loader (shuffled), seq_train_loader (unshuffled, for caching),
    and val_loader (unshuffled). All use wsss_collate_fn and pin_memory=True.
    """
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        collate_fn=wsss_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=wsss_collate_fn,
    )
    return train_loader, val_loader


def evaluate_masks(preds: torch.Tensor, targets: torch.Tensor, num_classes: int = 21, ignore_index: int = 255) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Computes per-class IoU and mIoU.
    preds: (B, H, W) or (H, W) tensor of predicted class indices.
    targets: (B, H, W) or (H, W) tensor of ground truth class indices.
    Returns:
        intersection: (num_classes,) tensor of intersection counts.
        union: (num_classes,) tensor of union counts.
        miou: Mean IoU over all valid classes present in the targets.
    """
    if preds.shape != targets.shape:
        raise ValueError(f"Preds shape {preds.shape} != targets shape {targets.shape}")
        
    device = preds.device
    intersection = torch.zeros(num_classes, device=device)
    union = torch.zeros(num_classes, device=device)
    
    valid = targets != ignore_index
    for c in range(num_classes):
        pred_c = (preds == c) & valid
        target_c = (targets == c) & valid
        intersection[c] = (pred_c & target_c).sum()
        union[c] = (pred_c | target_c).sum()
        
    return intersection, union


@torch.no_grad()
def cache_dino_tokens(
    dino_model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    loader: DataLoader,
    device: torch.device,
    embed_dim: int = 1024,   # Overridden automatically based on the model if needed
    dtype=torch.float16,
    desc="Caching DINO"
):
    """
    Helper to run a dataset through frozen DINOv3 model exactly once, returning a fully
    populated (N, NUM_PATCHES, EMBED_DIM) torch.Tensor cache and an (N, NUM_CLASSES) label tensor.
    """
    n = len(dataset)
    # Default cache structure unless we dynamically scale it
    dino_model.eval()
    
    cache_tensor = None
    labels_tensor = torch.zeros((n, 20), dtype=torch.float32, device=device)

    for indices, images, label_vec, _ in tqdm(loader, desc=desc):
        images = images.to(device, non_blocking=True)
        feats = dino_model.forward_features(images)["x_norm_patchtokens"].to(dtype)

        if cache_tensor is None:
            # Dynamically infer from the first batch
            _, num_patches, dynamic_embed_dim = feats.shape
            cache_tensor = torch.zeros((n, num_patches, dynamic_embed_dim), dtype=dtype, device=device)
            
        cache_tensor[indices] = feats
        labels_tensor[indices] = label_vec.to(device)

    mem_gb = cache_tensor.element_size() * cache_tensor.nelement() / (1024 ** 3)
    print(f"{desc}: Cached {n} samples ({mem_gb:.2f} GB on {device}).")
    
    return cache_tensor, labels_tensor



