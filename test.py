"""
Single Decoder WSSS: DINOv3 + SegDecoder only.

No pseudo masks. No CRF. No 3-stage pipeline.
One model, trained end-to-end with image-level labels only.

Architecture:
    Image [B, 3, H, W]
        → DINOv3 (frozen)
        → patch tokens [B, N, 384]
        → SegDecoder (4x upsample)
        → seg_map [B, 20, H, W]
        → GAP → logits [B, 20]
        → BCE loss vs image-level labels

Inference:
    seg_map → normalize → threshold background → mask [B, H, W]
"""

import os
import time
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import xml.etree.ElementTree as ET

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from huggingface_hub import login
from transformers import AutoModel


# ---------------------------------------------------------------------------
# Config — edit these instead of CLI args
# ---------------------------------------------------------------------------

VOC_ROOT     = 'data/VOCdevkit/VOC2012'
OUTPUT_DIR   = 'outputs/single_decoder'
IMG_SIZE     = 224
BATCH_SIZE   = 32
NUM_WORKERS  = 4
EPOCHS       = 30
LR           = 1e-3
WEIGHT_DECAY = 1e-4
BG_THRESHOLD = 0.30   # tune between 0.25–0.45 at inference


# ---------------------------------------------------------------------------
# VOC Classes
# ---------------------------------------------------------------------------

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(VOC_CLASSES)}

# One RGB color per class (plus background at index 0), used to colorize
# predicted masks for visualization. Follows the standard Pascal VOC palette.
VOC_COLORMAP = np.array([
    [0,0,0],[128,0,0],[0,128,0],[128,128,0],[0,0,128],
    [128,0,128],[0,128,128],[128,128,128],[64,0,0],[192,0,0],
    [64,128,0],[192,128,0],[64,0,128],[192,0,128],[64,128,128],
    [192,128,128],[0,64,0],[128,64,0],[0,192,0],[128,192,0],
    [0,64,128],
], dtype=np.uint8)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VOCDataset(Dataset):
    """
    Pascal VOC 2012.
    Returns image + multi-hot label vector [20] from XML annotations.
    """

    def __init__(self, root, split='train'):
        self.img_dir = os.path.join(root, 'JPEGImages')
        self.ann_dir = os.path.join(root, 'Annotations')

        split_file = os.path.join(root, 'ImageSets', 'Segmentation', f'{split}.txt')
        with open(split_file) as f:
            all_ids = [l.strip() for l in f]

        self.ids = [i for i in all_ids
                    if os.path.exists(os.path.join(self.ann_dir, f'{i}.xml'))]

        self.transform = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        print(f"[VOCDataset] {split}: {len(self.ids)} images")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        img = Image.open(os.path.join(self.img_dir, f'{img_id}.jpg')).convert('RGB')
        img = self.transform(img)

        label = torch.zeros(20, dtype=torch.float32)
        tree = ET.parse(os.path.join(self.ann_dir, f'{img_id}.xml'))
        for obj in tree.getroot().findall('object'):
            name = obj.find('name').text
            if name in CLASS_TO_IDX:
                label[CLASS_TO_IDX[name]] = 1.0

        return img, label


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SegDecoder(nn.Module):
    """
    Upsampling decoder: patch tokens → full-resolution seg map.

    DINOv3 outputs patch tokens at 1/14th the input resolution (e.g. 16×16
    patches for a 224×224 image → 16×16 tokens). We need to upsample back to
    the original image resolution so each pixel gets a class score — the same
    spatial resolution the ground-truth masks live at. Without upsampling, GAP
    over coarse 16×16 maps would still work for the classification loss, but
    inference masks would be too low-resolution to be useful.

    Four bilinear 2× blocks: 16 → 32 → 64 → 128 → 224 (≈ full resolution).
    """

    def __init__(self, embed_dim=384, num_classes=20, hidden_dim=256):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )

        self.up1 = self._up_block(hidden_dim,       hidden_dim)
        self.up2 = self._up_block(hidden_dim,       hidden_dim // 2)
        self.up3 = self._up_block(hidden_dim // 2,  hidden_dim // 4)
        self.up4 = self._up_block(hidden_dim // 4,  hidden_dim // 4)

        self.head = nn.Conv2d(hidden_dim // 4, num_classes, kernel_size=1)

    def _up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, patch_tokens):
        B, N, C = patch_tokens.shape
        h = w = int(N ** 0.5)

        x = patch_tokens.permute(0, 2, 1).reshape(B, C, h, w)
        x = self.proj(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.head(x)
        return x


class SingleDecoderWSS(nn.Module):
    """
    DINOv3 (frozen) + SegDecoder (trainable).

    Training:  seg_map → GAP → logits → BCE vs image labels
    Inference: seg_map → normalize → threshold → segmentation mask
    """

    def __init__(self, num_classes=20, hidden_dim=256):
        super().__init__()

        load_dotenv()
        hf_token = os.getenv("HF_TOKEN")
        login(token=hf_token)

        self.encoder = AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
        embed_dim = self.encoder.config.hidden_size

        for p in self.encoder.parameters():
            p.requires_grad = False

        self.decoder = SegDecoder(
            embed_dim=embed_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
        )

        print(f"[SingleDecoderWSS] DINOv3 ViT-S16 | embed_dim={embed_dim}")

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            seg_map: [B, 20, H, W]
            logits:  [B, 20]
        """
        out          = self.encoder(pixel_values=x)
        patch_tokens = out.last_hidden_state[:, 1:, :]  # drop [CLS] token

        seg_map = self.decoder(patch_tokens)
        seg_map = F.interpolate(seg_map, size=x.shape[2:],
                                mode='bilinear', align_corners=False)

        logits = F.adaptive_avg_pool2d(seg_map, 1).flatten(1)
        return seg_map, logits

    @torch.no_grad()
    def predict_mask(self, x):
        """
        Returns a segmentation mask [B, H, W]:
          0          → background
          1–20       → foreground class (1-indexed to keep 0 for bg)

        Steps:
          1. Forward pass → seg_map [B, 20, H, W]
          2. Zero-out classes the classifier says are absent (sigmoid < 0.2)
          3. Min-max normalize each class map to [0, 1]
          4. Pixels where all scores < BG_THRESHOLD → background (0)
          5. Otherwise → argmax + 1
        """
        self.eval()
        seg_map, logits = self.forward(x)

        present = (torch.sigmoid(logits) > 0.2).float()
        seg_map = seg_map * present.unsqueeze(-1).unsqueeze(-1)

        B, C, H, W = seg_map.shape
        flat    = seg_map.view(B, C, -1)
        s_min   = flat.min(-1).values[:, :, None, None]
        s_max   = flat.max(-1).values[:, :, None, None]
        seg_map = (seg_map - s_min) / (s_max - s_min + 1e-6)

        is_bg        = seg_map.max(dim=1).values < BG_THRESHOLD
        pred         = seg_map.argmax(dim=1) + 1
        pred[is_bg]  = 0
        return pred


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    train_ds = VOCDataset(VOC_ROOT, split='train')
    val_ds   = VOCDataset(VOC_ROOT, split='val')

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model     = SingleDecoderWSS().to(device)
    optimizer = AdamW(model.decoder.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    best_f1 = 0.0

    for epoch in range(1, EPOCHS + 1):
        # --- Train ---
        model.train()
        model.encoder.eval()  # keep encoder BN in eval mode
        t0 = time.time()

        for step, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)

            _, logits = model(imgs)
            loss      = F.binary_cross_entropy_with_logits(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
            optimizer.step()

            if step % 50 == 0:
                print(f"  [Ep {epoch}] step {step}/{len(train_loader)} | loss={loss.item():.4f}")

        scheduler.step()

        # --- Validate ---
        model.eval()
        val_loss, all_logits, all_labels = 0.0, [], []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                _, logits    = model(imgs)
                val_loss    += F.binary_cross_entropy_with_logits(logits, labels).item()
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        preds = (torch.sigmoid(all_logits) > 0.5).float()
        tp = (preds * all_labels).sum(0)
        fp = (preds * (1 - all_labels)).sum(0)
        fn = ((1 - preds) * all_labels).sum(0)
        p  = (tp / (tp + fp + 1e-6)).mean().item()
        r  = (tp / (tp + fn + 1e-6)).mean().item()
        f1 = 2 * p * r / (p + r + 1e-6)

        print(f"\n[Epoch {epoch}/{EPOCHS}] ({time.time()-t0:.1f}s) | "
              f"val_loss={val_loss/len(val_loader):.4f}  F1={f1:.4f}  P={p:.4f}  R={r:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_f1': best_f1,
            }, os.path.join(OUTPUT_DIR, 'best_model.pth'))
            print(f"  ★ New best F1={best_f1:.4f} saved\n")
        else:
            print()

    print(f"Done. Best val F1: {best_f1:.4f}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer(image_path, checkpoint_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SingleDecoderWSS().to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt['epoch']}, F1={ckpt['val_f1']:.4f})")

    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    img   = Image.open(image_path).convert('RGB')
    img_t = transform(img).unsqueeze(0).to(device)
    mask  = model.predict_mask(img_t).squeeze(0).cpu().numpy()  # [H, W]

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    img_np  = np.array(img.resize((IMG_SIZE, IMG_SIZE)))
    colored = VOC_COLORMAP[mask]
    present = np.unique(mask)
    classes_with_bg = ['background'] + VOC_CLASSES
    patches = [mpatches.Patch(color=VOC_COLORMAP[c] / 255., label=classes_with_bg[c])
               for c in present]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img_np);              axes[0].set_title('Input');       axes[0].axis('off')
    axes[1].imshow(img_np);              axes[1].imshow(colored, alpha=0.6)
    axes[1].set_title(f'Prediction (bg_thr={BG_THRESHOLD})')
    axes[1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig('inference_result.png', dpi=100, bbox_inches='tight')
    print("Saved inference_result.png")


if __name__ == '__main__':
    train()