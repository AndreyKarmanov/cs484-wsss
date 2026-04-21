"""CAM-based Prompting Module (CPM), ported from S2C (CVPR 2024).

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

    pgt: np.ndarray  # (H, W) uint8, class index in [0, num_classes-1] for fg, num_classes for bg
    score: np.ndarray  # (H, W) float32, aggregated confidence (0 where bg)
    masks: dict[int, np.ndarray]  # {class_idx: (H, W) bool} the chosen SAM mask per class
    confs: dict[int, float]  # {class_idx: float} SAM mask-quality score for the chosen mask
    points: dict[int, np.ndarray]  # {class_idx: (N, 2) int xy} positive prompts used


def cam_to_cpm_points(
    cam: np.ndarray,
    th_multi: float = 0.5,
    min_distance: int = 20,
    max_filter_size: int = 3,
) -> np.ndarray:
    """S2C's local-peak prompt extraction for one (H, W) CAM in [0, 1].

    Returns positive point prompts in image coords as an (N, 2) int array of
    `(x, y)` pairs. Always includes the global argmax. Local peaks below
    `th_multi` are dropped.

    Mirrors the "Sample local peaks" block of S2C's `model_s2c.py`. Notable
    quirks preserved verbatim from the reference:

      * The 3x3 maximum filter is applied to the CAM before peak finding.
      * `peaks_valid[1:]` drops the first local peak before concatenating with
        the global argmax (the global argmax is otherwise duplicated).
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
    """Run S2C-style CPM on one image: CAMs -> point prompts -> SAM2 -> aggregate.

    Args:
        image: either a path, a PIL.Image, or an (H, W, 3) uint8 numpy array.
            Forwarded to `predictor.set_image`.
        cams: `{class_idx: (H, W) float}` per-class CAMs. Should already be in
            the original image's pixel space (matches your existing
            `result["dino_refined"]` format). Each CAM is internally re-normalized
            to [0, 1] by dividing by its max, like S2C's `cam_ms` step.
        predictor: a SAM 2 `SAM2ImagePredictor` (already constructed).
        num_classes: total number of foreground classes (VOC: 20). The output
            `pgt` uses this value as the background label.
        th_multi: minimum normalized-CAM value for a local peak to count
            as a positive prompt (S2C default 0.5).
        min_distance: NMS radius for local peaks in CAM pixels (S2C default 20).
        idx_max_sam: which of SAM's 3 multimask outputs to take. S2C found
            index 2 (the largest/most-inclusive) to work best empirically.

    Returns:
        `CPMResult`. The `pgt` field is the headline output: an (H, W) uint8
        class-index map where `pgt == num_classes` means background.
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

        # S2C-style confidence-based aggregation:
        # mean CAM inside the mask is constant per class; SAM gives one
        # confidence per mask, so the per-pixel score is also constant inside.
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
