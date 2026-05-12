# Abstract

Weakly supervised semantic segmentation (WSSS) aims to recover pixel-level masks from weak supervision such as image-level tags, but the lack of dense annotations makes the problem especially sensitive to how class evidence is localized and refined. In this project, we explore three WSSS pipelines on PASCAL VOC 2012: a frozen DINOv3 backbone with a linear CAM classifier, a DINOv3 + SAM2 distillation approach, and a training-free CLIP-ES + DINO affinity + SAM2 pipeline. The goal is to compare how far modern foundation-model representations can go when combined with lightweight segmentation heads or prompt-based refinement.

The methodology is organized around a progression from simple to more structured spatial reasoning. The basic model uses cached DINO patch tokens and a linear classifier to produce class activation maps, the distillation model adds a convolutional decoder and SAM2-generated pseudo-labels to encourage dense predictions, and the final pipeline combines CLIP-ES CAMs, DINO affinity refinement, and CPM-driven SAM2 masks. This setup makes it possible to compare not only final mIoU, but also the behavior of each stage: raw CAM quality, boundary refinement, and prompt-based mask generation.

Overall, the experiments show that stronger representation models help, but that the quality of refinement matters as much as the backbone itself. The simpler DINO-based methods produce understandable baselines, while the CLIP-ES + DINO + SAM2 pipeline offers the most interpretable multi-stage visualization but also reveals failure modes where refinement can hurt rather than help. These results suggest that WSSS performance depends not just on model strength, but on whether the refinement pipeline preserves useful class evidence while improving spatial alignment.

# Code Libraries

Below are the main libraries used in this project and why they matter.

- torch, torchvision
  - Used for model definition, training loops, tensor operations, and VOC dataset loading.
  - Core framework for all three architectures.

- sam2
  - Used for SAM2 mask prediction from prompts in Architecture 2 and Architecture 3.

- dinov3
  - Used as the frozen representation backbone for Architectures 1 and 2, and for affinity refinement in Architecture 3.

- clip_es and pytorch-grad-cam
  - Used in Architecture 3 to generate CLIP-based CAMs and attention-guided refinement.

Installation
- `pip install torch torchvision numpy matplotlib tqdm opencv-python pillow lxml scipy scikit-image torchmetrics termcolor sam2 ftfy regex ttach`

Download models and place into `weights/` folder
- [SAM2_tiny](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt)
- [SAM2_small](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt)
- [DINO](https://github.com/facebookresearch/dinov3#pretrained-models)
  - download the ViT-S/16 distilled and ViT-L/16 distilled
  - meta requires you to fill out a form before getting temporary links
  - run `git clone https://github.com/facebookresearch/dinov3` in directory with this notebook


Project libraries (mylibs) used:
- dataset.py: shared dataset wrappers, transforms, caching, and evaluation helpers.
- cpm.py: CPM prompt extraction and SAM integration for class-wise masks.

External libraries to be downloaded

Run following git clones from within the same folder that this notebook is in.
- dinov3/: backbone implementation and weights interface.
  - `git clone https://github.com/facebookresearch/dinov3`
- clip_es/: CLIP-ES pipeline and Grad-CAM utilities used in Architecture 3.
  - `git clone https://github.com/linyq2117/CLIP-ES clip_es`
