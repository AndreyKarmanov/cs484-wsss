"""CAM-based Prompting Module (CPM), from S2C (CVPR 2024).

Reference: Kweon & Yoon, "From SAM to CAMs: Exploring Segment Anything Model
for Weakly Supervised Semantic Segmentation," CVPR 2024.
Original code: https://github.com/sangrockEG/S2C/blob/main/models/model_s2c.py

What is CPM?
"CPM extracts prompts from the CAM of each class and uses them to 
generate class-specific segmentation masks through SAM."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.feature import peak_local_max


@dataclass
class CPMResult:
    """Output of `cpm_from_cams` for a single image."""

    pgt: np.ndarray
    score: np.ndarray
    masks: dict[int, np.ndarray]
    confs: dict[int, float]
    points: dict[int, np.ndarray]


def cam_to_cpm_points(
    cam: np.ndarray,
    th_multi: float = 0.5,
    min_distance: int = 20,
    max_filter_size: int = 3,
) -> np.ndarray:
    """S2C's local-peak prompt extraction for one (H, W) CAM in [0, 1].
    """
    H, W = cam.shape

    flat_argmax = int(np.argmax(cam))
    py, px = divmod(flat_argmax, W)
    peak_max = np.array([[py, px]], dtype=np.int64)

    cam_filtered = ndi.maximum_filter(cam, size=max_filter_size, mode="constant")
    peaks = peak_local_max(cam_filtered, min_distance=min_distance)
    if peaks.size:
        peaks_valid = peaks[cam[peaks[:, 0], peaks[:, 1]] > th_multi]
    else:
        peaks_valid = np.empty((0, 2), dtype=np.int64)

    if len(peaks_valid) > 1:
        all_peaks = np.concatenate([peak_max, peaks_valid[1:]], axis=0)
    else:
        all_peaks = peak_max

    points_xy = np.flip(all_peaks, axis=-1).astype(np.int64)
    return points_xy


def _normalize_cam(cam: np.ndarray) -> np.ndarray:
    cam = cam.astype(np.float32, copy=False)
    cmax = float(cam.max())
    if cmax <= 1e-5:
        return cam
    return cam / cmax


def cpm_from_cams(
    image: np.ndarray | Image.Image | str,
    cams: dict[int, np.ndarray],
    predictor,
    *,
    num_classes: int = 20,
    th_multi: float = 0.5,
    min_distance: int = 20,
    idx_max_sam: int = 2,
) -> CPMResult:
    """Run CPM on one image: CAMs -> point prompts -> SAM2 -> aggregate.
    """
    if isinstance(image, str):
        img_np = np.array(Image.open(image).convert("RGB"))
    elif isinstance(image, Image.Image):
        img_np = np.array(image.convert("RGB"))
    else:
        img_np = np.asarray(image)

    H, W = img_np.shape[:2]

    predictor.set_image(img_np)

    masks_per_class: dict[int, np.ndarray] = {}
    confs_per_class: dict[int, float] = {}
    points_per_class: dict[int, np.ndarray] = {}

    score_map = np.full((num_classes, H, W), -1e5, dtype=np.float32)

    for c, cam in cams.items():
        if cam.shape != (H, W):
            raise ValueError(
                f"CAM for class {c} has shape {cam.shape}, expected {(H, W)}. "
                "Resize to the original image size before calling cpm_from_cams."
            )

        cam_norm = _normalize_cam(cam)

        points_xy = cam_to_cpm_points(
            cam_norm,
            th_multi=th_multi,
            min_distance=min_distance,
        )
        point_labels = np.ones(len(points_xy), dtype=np.int32)

        masks, scores, _ = predictor.predict(
            point_coords=points_xy.astype(np.float32),
            point_labels=point_labels,
            multimask_output=True,
        )
        idx = min(idx_max_sam, masks.shape[0] - 1)
        target_mask = masks[idx].astype(bool)
        target_conf = float(scores[idx])

        masks_per_class[c] = target_mask
        confs_per_class[c] = target_conf
        points_per_class[c] = points_xy

        if target_mask.any():
            mean_cam_inside = float(cam_norm[target_mask].mean())
            agg_score = target_conf * mean_cam_inside
            score_map[c, target_mask] = agg_score

    pgt = score_map.argmax(axis=0).astype(np.uint8)
    pgt_score = score_map.max(axis=0)
    bg_pixels = pgt_score < 0
    pgt[bg_pixels] = num_classes
    pgt_score[bg_pixels] = 0.0

    return CPMResult(
        pgt=pgt,
        score=pgt_score.astype(np.float32),
        masks=masks_per_class,
        confs=confs_per_class,
        points=points_per_class,
    )
