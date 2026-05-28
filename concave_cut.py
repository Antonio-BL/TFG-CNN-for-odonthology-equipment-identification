# concave_cut.py
# Geometric cut of fused instruments at the best concave point.
#
# Pipeline role
# ─────────────
#   1. select_best_concave_points — pick the candidate with max chord_len
#      per outlier bbox.  A large chord length corresponds to a wide,
#      pronounced concavity, which is the most reliable place to cut a
#      pair of fused instruments.
#   2. apply_concave_cuts — draw a black VERTICAL line through that point
#      on a copy of the binary mask.  The line is aligned with the image
#      y-axis regardless of bbox orientation, and it is long enough to
#      fully cross the bbox at any rotation.
#
# After this step, re-running segmentation.find_bboxes on the cut mask
# yields one bbox per separated instrument and gives the final count.
#
# Architecture: zero matplotlib imports.  Returns data only.

from __future__ import annotations

import numpy as np
import cv2 as cv

from config import PreprocessConfig


def _concave_grade(pt: dict, pts: list[dict]) -> float:
    """Compute the weighted grade of a concave point relative to its peers.

    Grade = 0.20 · norm(chord_len)
           + 0.50 · norm(kappa)
           + 0.30 · norm(roi_kappa_mean)

    Each property is min-max normalised across all candidates in `pts` so
    that the three quantities (which live on different scales) contribute
    proportionally to their weights.  When all candidates share the same
    value for a property, that term contributes equally to everyone and the
    tie is broken by the remaining properties.

    Args:
        pt:  The candidate whose grade is requested.
        pts: Full list of candidates for this bbox (used for normalisation).

    Returns:
        float grade in [0, 1].
    """
    def _norm(key: str) -> float:
        values = [p[key] for p in pts]
        lo, hi = min(values), max(values)
        if hi - lo < 1e-12:
            return 1.0   # all equal — full score to everyone
        return (pt[key] - lo) / (hi - lo)

    return 0.20 * _norm('chord_len') + 0.50 * _norm('kappa') + 0.30 * _norm('roi_kappa_mean')


def select_best_concave_points(
    concave_pts: dict[int, list[dict]],
) -> dict[int, dict | None]:
    """Pick the best concave point per outlier bbox using a weighted grade.

    Grade weights:
        20 %  chord_len       — width of the concave notch
        50 %  kappa           — sharpness of the bend at the POI vertex
        30 %  roi_kappa_mean  — average curvature across the high-curvature ROI

    All three properties are min-max normalised across the candidates of the
    same bbox before weighting, so scale differences do not bias the result.

    Args:
        concave_pts: dict[bbox_idx → list of point dicts] from
                     detect_concave_points.

    Returns:
        dict[bbox_idx → best point dict OR None when the input list was
        empty].  Key set matches the input.
    """
    best: dict[int, dict | None] = {}
    for idx, pts in concave_pts.items():
        if not pts:
            best[idx] = None
        else:
            best[idx] = max(pts, key=lambda p: _concave_grade(p, pts))
    return best


def apply_concave_cuts(
    seg_binary: np.ndarray,
    outlier_bboxes: list[tuple],
    best_points: dict[int, dict | None],
    cfg: PreprocessConfig,
) -> np.ndarray:
    """Return a copy of seg_binary with vertical cut lines drawn at the best
    concave point of each outlier bbox.

    Per outlier bbox with a non-None best point, draws a black line:
      - centred on the concave point (pt['x'], pt['y']),
      - VERTICAL in image coordinates (constant x, varying y) regardless
        of the bbox orientation,
      - thickness = cfg.concave_cut_line_width pixels,
      - extended well beyond the bbox to guarantee full crossing,
      - pixel value = 0 (background).

    Args:
        seg_binary:     uint8 (H, W) binary mask; instruments = 255.
        outlier_bboxes: List of (center, size, angle, area) tuples;
                        same indexing as best_points (0-based outlier idx).
        best_points:    Output of select_best_concave_points.
        cfg:            PreprocessConfig (reads concave_cut_line_width).

    Returns:
        uint8 (H, W) binary mask with cut lines drawn.  The input is
        not modified.
    """
    cut_mask   = seg_binary.copy()
    line_width = cfg.concave_cut_line_width

    for bbox_idx, bbox in enumerate(outlier_bboxes):
        pt = best_points.get(bbox_idx)
        if pt is None:
            continue

        _center, size, _angle, _area = bbox
        w, h = size

        # Vertical line in image space: x is fixed at the concave point,
        # y extends in both directions.  Using the bbox diagonal as the
        # half-length guarantees the line spans the bbox at any rotation.
        # Drawing extra pixels outside the bbox is harmless — the mask is
        # already 0 there.
        half_len = int(np.hypot(w, h) * 0.6)

        cx, cy = int(pt['x']), int(pt['y'])
        cv.line(
            cut_mask,
            (cx, cy - half_len),
            (cx, cy + half_len),
            color=0, thickness=line_width,
        )

    return cut_mask
