from torchvision.datasets import VOCSegmentation
import os

DATA_ROOT = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_ROOT, exist_ok=True)

for split in ('train', 'val'):
    print(f"Downloading VOC2012 {split} split...")
    VOCSegmentation(root=DATA_ROOT, year='2012', image_set=split, download=True)

print(f"\nDone. Dataset is at: {os.path.join(DATA_ROOT, 'VOCdevkit', 'VOC2012')}")
print(f"Use --voc_root {os.path.join(DATA_ROOT, 'VOCdevkit', 'VOC2012')}")
