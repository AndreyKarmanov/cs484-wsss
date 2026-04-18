Weakly-supervised semantic segmentation (WSSS): with seeds, boxes, or image-level tags. Feel free to explore and compare any approaches to the problem. You can explore loss functions directly using seeds/boxes or tags during training. You can use ImageNet pretrained encoder, but this works much better with (fixed) foundation model representation like DINO. It is enough to use a linear classifier (or some simple 2-3 layer decoder) on top of DINO representation to make things work well. Feel free to implement standard methods or experiemnt with your own ideas or modifications. In either case, you should clearly explain and motivate your approach.


[Download DINO](https://github.com/facebookresearch/dinov3#pretrained-models)
Download one of the ViT LVD-1689M models (e.g., `ViT-S/16 distilled`) and place it in the project directory.

[Download SAM](https://github.com/facebookresearch/sam2#getting-started)
Download any of the SAM checkpoints 

You will have to adjust the paths in `unified.ipynb`
