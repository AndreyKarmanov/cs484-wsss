"""
Inference on VOC 2012 val split.

Loads a trained SingleDecoderWSS checkpoint, runs predict_mask() on every
val image, and reports:
  - Overall mIoU (21 classes including background)
  - Per-class IoU breakdown
  - Saved predicted mask PNGs  →  outputs/masks/
  - Side-by-side visualizations  →  outputs/vis/
"""

import os
import numpy as np
from PIL import Image
from tqdm import tqdm

import random
from multiprocessing import Pool, cpu_count

import matplotlib
matplotlib.use('Agg')  # non-interactive backend — prevents hangs in Colab
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import torchvision.transforms as T

from torch.utils.data import Dataset, DataLoader

# Reuse everything defined in the training script
from test import (
    SingleDecoderWSS,
    VOC_CLASSES,
    VOC_COLORMAP,
    VOC_ROOT,
    IMG_SIZE,
    BG_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHECKPOINT   = 'outputs/single_decoder/best_model.pth'
OUT_MASKS    = 'outputs/masks'
OUT_VIS      = 'outputs/vis'
NUM_CLASSES  = 21    # 0=background, 1-20=VOC classes
SAVE_VIS        = True # set True to save side-by-side visualizations
SAVE_MASKS      = False # set True to save raw predicted mask PNGs
VIS_SAMPLE_SIZE = 50    # how many random images to visualize
BATCH_SIZE   = 32    # match training batch size — keeps GPU utilization high
NUM_WORKERS  = 4     # parallel image loading
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------------------
# Dataset (val images + GT masks)
# ---------------------------------------------------------------------------

class ValDataset(Dataset):
    """
    Loads val images and GT masks together so a DataLoader can prefetch
    and transform in parallel worker processes while the GPU runs inference.
    """

    def __init__(self):
        split_file = os.path.join(VOC_ROOT, 'ImageSets', 'Segmentation', 'val.txt')
        with open(split_file) as f:
            self.ids = [l.strip() for l in f]

        self.img_transform = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        img    = Image.open(os.path.join(VOC_ROOT, 'JPEGImages', f'{img_id}.jpg')).convert('RGB')
        img_t  = self.img_transform(img)
        img_np = np.array(img.resize((IMG_SIZE, IMG_SIZE)))  # for visualization

        # GT mask — 255 = ignore boundary
        gt = np.array(Image.open(
            os.path.join(VOC_ROOT, 'SegmentationClass', f'{img_id}.png')
        ).resize((IMG_SIZE, IMG_SIZE), Image.NEAREST))

        return img_id, img_t, img_np, gt


# ---------------------------------------------------------------------------
# IoU helpers
# ---------------------------------------------------------------------------

def compute_iou(pred, gt, num_classes=NUM_CLASSES):
    """
    Compute per-class IoU and mean IoU, ignoring pixels labelled 255.

    Returns:
        iou_per_class: np.array [num_classes]  — NaN for absent classes
        miou: float
    """
    valid = gt != 255
    pred, gt = pred[valid], gt[valid]

    iou_per_class = np.full(num_classes, np.nan)
    for c in range(num_classes):
        tp = np.logical_and(pred == c, gt == c).sum()
        fp = np.logical_and(pred == c, gt != c).sum()
        fn = np.logical_and(pred != c, gt == c).sum()
        denom = tp + fp + fn
        if denom > 0:
            iou_per_class[c] = tp / denom

    miou = np.nanmean(iou_per_class)
    return iou_per_class, miou


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_vis(img_id, img_np, pred_mask, gt_mask):
    colored_pred = VOC_COLORMAP[pred_mask]
    colored_gt   = VOC_COLORMAP[np.where(gt_mask == 255, 0, gt_mask)]  # map ignore→bg for display

    classes_with_bg = ['background'] + VOC_CLASSES
    present = np.unique(pred_mask)
    patches = [mpatches.Patch(color=VOC_COLORMAP[c] / 255., label=classes_with_bg[c])
               for c in present]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(img_np);                        axes[0].set_title('Input');          axes[0].axis('off')
    axes[1].imshow(colored_gt);                    axes[1].set_title('GT Mask');        axes[1].axis('off')
    axes[2].imshow(img_np); axes[2].imshow(colored_pred, alpha=0.6)
    axes[2].set_title(f'Prediction (bg_thr={BG_THRESHOLD})')
    axes[2].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    axes[2].axis('off')

    plt.suptitle(img_id, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_VIS, f'{img_id}.png'), dpi=80, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Multiprocessing worker (must be top-level for pickling)
# ---------------------------------------------------------------------------

def _save_vis_worker(args):
    img_id, img_np, pred_mask, gt_mask = args
    save_vis(img_id, img_np, pred_mask, gt_mask)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_MASKS, exist_ok=True)
    os.makedirs(OUT_VIS,   exist_ok=True)

    print(f"Device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: No GPU detected — running on CPU.")

    # Load model
    model = SingleDecoderWSS().to(DEVICE)
    ckpt  = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint — epoch {ckpt['epoch']}, val F1={ckpt['val_f1']:.4f}\n")

    # DataLoader — workers prefetch+transform while GPU runs inference
    loader = DataLoader(ValDataset(), batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
    print(f"Running on {len(loader.dataset)} val images (batch={BATCH_SIZE})...\n")

    all_iou  = []
    mask_buf = {}
    vis_buf  = {}

    for img_ids, imgs, imgs_np, gt_masks in tqdm(loader):
        imgs = imgs.to(DEVICE, non_blocking=True)  # non_blocking works with pin_memory

        with torch.no_grad():
            seg_map, logits = model(imgs)

            # Suppress absent classes, normalize, threshold background
            present = (torch.sigmoid(logits) > 0.2).float()
            seg_map = seg_map * present.unsqueeze(-1).unsqueeze(-1)
            B, C, H, W = seg_map.shape
            flat    = seg_map.view(B, C, -1)
            s_min   = flat.min(-1).values[:, :, None, None]
            s_max   = flat.max(-1).values[:, :, None, None]
            seg_map = (seg_map - s_min) / (s_max - s_min + 1e-6)
            is_bg   = seg_map.max(dim=1).values < BG_THRESHOLD
            pred    = seg_map.argmax(dim=1) + 1
            pred[is_bg] = 0
            pred_masks = pred.cpu().numpy()   # [B, H, W]

        gt_masks = gt_masks.numpy()           # [B, H, W]
        imgs_np  = imgs_np.numpy()            # [B, H, W, 3]

        for i, img_id in enumerate(img_ids):
            iou_per_class, _ = compute_iou(pred_masks[i], gt_masks[i])
            all_iou.append(iou_per_class)

            if SAVE_MASKS:
                mask_buf[img_id] = pred_masks[i]
            if SAVE_VIS and len(vis_buf) < VIS_SAMPLE_SIZE:
                vis_buf[img_id] = (imgs_np[i], pred_masks[i], gt_masks[i])

    # ---------------------------------------------------------------------------
    # Batched writes
    # ---------------------------------------------------------------------------

    if SAVE_MASKS:
        print(f"\nWriting {len(mask_buf)} mask PNGs...")
        for img_id, mask in tqdm(mask_buf.items()):
            Image.fromarray(mask.astype(np.uint8)).save(
                os.path.join(OUT_MASKS, f'{img_id}.png'))

    if SAVE_VIS:
        print(f"\nWriting {len(vis_buf)} visualizations across {cpu_count()} workers...")
        args = [(img_id, img_np, pred_mask, gt_mask)
                for img_id, (img_np, pred_mask, gt_mask) in vis_buf.items()]
        with Pool(processes=cpu_count()) as pool:
            list(tqdm(pool.imap(_save_vis_worker, args), total=len(args)))

    # ---------------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------------

    all_iou   = np.stack(all_iou)                        # [N, 21]
    class_iou = np.nanmean(all_iou, axis=0)              # [21]  — mean over images
    miou      = np.nanmean(class_iou)

    classes_with_bg = ['background'] + VOC_CLASSES
    col_w = max(len(c) for c in classes_with_bg) + 2

    print("\n" + "=" * 45)
    print(f"{'Class':<{col_w}}  IoU")
    print("-" * 45)
    for i, (name, iou) in enumerate(zip(classes_with_bg, class_iou)):
        marker = "  ← best" if iou == np.nanmax(class_iou) else ""
        print(f"{name:<{col_w}}  {iou:.4f}{marker}")
    print("=" * 45)
    print(f"{'mIoU':<{col_w}}  {miou:.4f}")
    print("=" * 45)
    print(f"\nMasks saved to  : {OUT_MASKS}/")
    print(f"Visuals saved to: {OUT_VIS}/")


if __name__ == '__main__':
    main()