# concave_points.py
# Concave-point detection on instrument outer contours.
#
# Implements Section 3 of Miró-Nicolau et al. (2024):
#   1. RDP simplification of the outer contour.
#   2. k-curvature estimation at each vertex.
#   3. Recursive ROI finding on the curvature signal.
#   4. Weighted-median point-of-interest selection inside each ROI.
#   5. Chord-midpoint concavity test.
#
# Public API:
#   detect_concave_points(seg_binary, outlier_bboxes, cfg)
#       → dict[int, list[dict]]
#
#   Each dict in the list has:
#       'x'              int    full-image pixel x
#       'y'              int    full-image pixel y
#       'kappa'          float  k-curvature at the POI vertex
#       'roi_len'        int    simplified-vertex count in the ROI
#       'roi_kappa_mean' float  mean curvature across the ROI
#       'roi_kappa_max'  float  peak curvature in the ROI
#       'chord_len'      float  Euclidean distance between the ±k neighbors
#
# Architecture: ZERO matplotlib imports. Returns data only.

from __future__ import annotations

import numpy as np
import cv2 as cv

from config import PreprocessConfig


# ================================================================== #
#  Step 1 — RDP contour simplification                              #
# ================================================================== #

def rdp_simplify(contour: np.ndarray, epsilon: float) -> np.ndarray:
    """Simplify a closed contour with the Ramer-Douglas-Peucker algorithm.

    Args:
        contour: (N, 1, 2) int32 array as returned by cv.findContours.
        epsilon: Maximum distance between the original curve and the
                 approximation.  Larger values → fewer points.

    Returns:
        (M, 2) float64 array of simplified vertices (M ≤ N).
    """
    approx = cv.approxPolyDP(contour, epsilon=epsilon, closed=True)
    return approx.reshape(-1, 2).astype(np.float64)


# ================================================================== #
#  Step 2 — k-curvature                                             #
# ================================================================== #

def compute_k_curvature(contour: np.ndarray, k: int) -> np.ndarray:
    """Estimate turning angle at each vertex using k-step neighbours.

    For vertex i the angle is formed by vectors:
        v1 = contour[i]       − contour[(i−k) % N]
        v2 = contour[(i+k)%N] − contour[i]

    The curvature value κ[i] = arccos(clip(v1̂ · v2̂, −1, 1)) ∈ [0, π].
    A high value means the contour bends sharply at i.

    Args:
        contour: (N, 2) array of 2-D vertex coordinates.
        k:       Look-ahead/look-behind step.

    Returns:
        (N,) float64 array of curvature values in [0, π].
    """
    contour = np.asarray(contour, dtype=np.float64)
    N = len(contour)
    kappa = np.zeros(N, dtype=np.float64)

    for i in range(N):
        prev = contour[(i - k) % N]
        curr = contour[i]
        nxt  = contour[(i + k) % N]

        v1 = curr - prev
        v2 = nxt  - curr

        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            kappa[i] = 0.0
            continue

        dot = np.dot(v1 / n1, v2 / n2)
        kappa[i] = np.arccos(np.clip(dot, -1.0, 1.0))

    return kappa


# ================================================================== #
#  Step 3 — Recursive ROI finding                                   #
# ================================================================== #

def _find_rois_recursive(
    curvature: np.ndarray,
    t: float,
    lmin: int,
    lmax: int,
    delta_t: float,
    k: int,
    max_depth: int = 20,
) -> list[tuple[int, int]]:
    """Find ROIs (runs of high curvature) recursively.

    A ROI is a maximal contiguous run (circular) where κ > t.

    Rules:
        Case 1 — lmin ≤ len ≤ lmax  →  accept the ROI.
        Case 2 — len > lmax          →  recurse on that run with t + delta_t.
        Case 3 — len < lmin          →  try to merge with the closest neighbour.
            3a — merged len ≥ lmin   →  recurse on the merged run with t + delta_t.
            3b — merged len < lmin   →  skip (too short even after merge).

    Args:
        curvature:  (N,) curvature array.
        t:          Current threshold.
        lmin:       Minimum ROI length (inclusive).
        lmax:       Maximum ROI length (inclusive).
        delta_t:    Threshold increment for recursion.
        k:          k-curvature parameter (kept for signature symmetry).
        max_depth:  Recursion depth guard.

    Returns:
        List of (start, end) index pairs (inclusive, circular).
        Indices are into the original `curvature` array.
    """
    if max_depth <= 0:
        return []

    N = len(curvature)
    above = curvature > t

    if not np.any(above):
        return []

    # Find the first False so that we never split a wrap-around run
    first_false = 0
    for f in range(N):
        if not above[f]:
            first_false = f
            break
    else:
        # All above → one giant run
        if N <= lmax:
            return [(0, N - 1)]
        return _find_rois_recursive(
            curvature, t + delta_t, lmin, lmax, delta_t, k, max_depth - 1
        )

    doubled = np.concatenate([above, above])
    runs_sl: list[tuple[int, int]] = []   # (start_in_original, length)

    i = first_false
    end_scan = first_false + N
    while i < end_scan:
        if doubled[i % N]:
            s = i % N
            j = i
            while j < end_scan and doubled[j % N]:
                j += 1
            runs_sl.append((s, j - i))
            i = j
        else:
            i += 1

    rois: list[tuple[int, int]] = []
    for run_idx, (start, length) in enumerate(runs_sl):
        if lmin <= length <= lmax:
            rois.append((start, (start + length - 1) % N))

        elif length > lmax:
            sub_curv = np.array([curvature[(start + j) % N] for j in range(length)])
            for ss, se in _find_rois_recursive(
                sub_curv, t + delta_t, lmin, lmax, delta_t, k, max_depth - 1,
            ):
                rois.append(((start + ss) % N, (start + se) % N))

        else:
            # Case 3: too short
            if len(runs_sl) < 2:
                continue

            prev_idx = (run_idx - 1) % len(runs_sl)
            next_idx = (run_idx + 1) % len(runs_sl)
            prev_start, prev_len = runs_sl[prev_idx]
            next_start, next_len = runs_sl[next_idx]
            prev_end = (prev_start + prev_len - 1) % N
            curr_end = (start + length - 1) % N
            gap_prev = (start - prev_end - 1) % N
            gap_next = (next_start - curr_end - 1) % N

            if gap_prev <= gap_next:
                partner_start, partner_len = prev_start, prev_len
            else:
                partner_start, partner_len = next_start, next_len

            merged_len = length + partner_len
            if merged_len >= lmin:
                all_idx = (
                    [(start + j) % N for j in range(length)]
                    + [(partner_start + j) % N for j in range(partner_len)]
                )
                sub_curv = np.array([curvature[i] for i in all_idx])
                for ss, se in _find_rois_recursive(
                    sub_curv, t + delta_t, lmin, lmax, delta_t, k, max_depth - 1,
                ):
                    rois.append((all_idx[ss], all_idx[se]))

    return rois


# ================================================================== #
#  Step 4 — Weighted-median point-of-interest selection             #
# ================================================================== #

def _select_point_of_interest(
    roi: tuple[int, int],
    curvature: np.ndarray,
) -> int:
    """Select the point of interest inside a ROI using a weighted median.

    The curvature values inside the ROI are normalised to [0, 1] and used
    as weights.  The selected index is the one where the cumulative weight
    first reaches or exceeds 0.5.

    Args:
        roi:       (start, end) inclusive index pair (circular).
        curvature: (N,) curvature array.

    Returns:
        Index into `curvature` of the selected point of interest.
    """
    N = len(curvature)
    start, end = roi
    roi_len = (end - start) % N + 1

    indices = [(start + j) % N for j in range(roi_len)]
    values  = np.array([curvature[i] for i in indices], dtype=np.float64)

    total = values.sum()
    if total < 1e-12:
        return indices[len(indices) // 2]

    weights = values / total
    cumsum  = np.cumsum(weights)

    median_pos = int(np.searchsorted(cumsum, 0.5))
    median_pos = min(median_pos, len(indices) - 1)
    return indices[median_pos]


# ================================================================== #
#  Step 5 — Chord-midpoint concavity test                           #
# ================================================================== #

def _is_concave(
    contour: np.ndarray,
    point_idx: int,
    seg_binary: np.ndarray,
    k: int,
) -> bool:
    """Return True if the contour point at point_idx is concave.

    The test draws a chord between the k-step neighbours of point_idx and
    checks whether the midpoint of that chord falls OUTSIDE the binary mask.
    If it does, the contour is locally concave at point_idx.

    Args:
        contour:    (N, 2) array of vertex coordinates (full-image space).
        point_idx:  Index of the candidate concave point.
        seg_binary: uint8 (H, W) binary mask; instruments = 255.
        k:          k-curvature look-around step.

    Returns:
        True if the midpoint of the chord lies outside the mask.
    """
    N = len(contour)
    p_prev = contour[(point_idx - k) % N]
    p_next = contour[(point_idx + k) % N]

    mx = int(round((p_prev[0] + p_next[0]) / 2.0))
    my = int(round((p_prev[1] + p_next[1]) / 2.0))

    h, w = seg_binary.shape
    mx = max(0, min(w - 1, mx))
    my = max(0, min(h - 1, my))

    return seg_binary[my, mx] == 0   # concave if midpoint is in background


# ================================================================== #
#  Public API                                                        #
# ================================================================== #

def detect_concave_points(
    seg_binary: np.ndarray,
    outlier_bboxes: list[tuple],
    cfg: PreprocessConfig,
) -> dict[int, list[dict]]:
    """Detect concave points on instrument outer contours inside outlier bboxes.

    For each outlier bbox:
      1. Crop binary mask to the axis-aligned bounding rect of oriented bbox.
      2. Mask the crop to the oriented bbox shape.
      3. Find the largest external contour in the crop.
      4. Simplify it with RDP (cp_rdp_epsilon).
      5. Skip if fewer than 2*k+1 points remain (warning printed).
      6. Compute k-curvature on the simplified contour.
      7. Find high-curvature ROIs via _find_rois_recursive.
      8. Select one POI per ROI via weighted median.
      9. Keep only points that pass the chord-midpoint concavity test.
      10. Convert to full-image coords.

    Args:
        seg_binary:     uint8 (H, W) binary mask; instruments = 255.
        outlier_bboxes: List of (center, size, angle, area) tuples.
        cfg:            PreprocessConfig with cp_* and seg_* parameters.

    Returns:
        Dict mapping 0-based outlier bbox index → list of concave-point dicts.
        Each dict contains:
            'x'              int    full-image pixel x
            'y'              int    full-image pixel y
            'kappa'          float  k-curvature at the POI vertex
            'roi_len'        int    simplified-vertex count in the ROI
            'roi_kappa_mean' float  mean curvature across the ROI
            'roi_kappa_max'  float  peak curvature in the ROI
            'chord_len'      float  Euclidean px distance between ±k neighbors
        Bboxes with no concave points map to an empty list.
    """
    result: dict[int, list[dict]] = {}

    h_img, w_img = seg_binary.shape
    k        = cfg.cp_k
    lmin     = cfg.cp_lmin
    lmax     = cfg.cp_lmax
    delta_t  = cfg.cp_delta_t
    eps      = cfg.cp_rdp_epsilon
    min_area = cfg.seg_min_contour_area
    min_pts  = 2 * k + 1

    for bbox_idx, bbox in enumerate(outlier_bboxes):
        result[bbox_idx] = []

        center, size, angle = bbox[0], bbox[1], bbox[2]
        box_pts = np.int32(cv.boxPoints((center, size, angle)))

        ax, ay, aw, ah = cv.boundingRect(box_pts)
        x1 = max(0, ax)
        y1 = max(0, ay)
        x2 = min(w_img, ax + aw)
        y2 = min(h_img, ay + ah)
        if x2 <= x1 or y2 <= y1:
            continue

        # Crop to axis-aligned bbox bounds
        roi_crop = seg_binary[y1:y2, x1:x2]

        # Mask to oriented bbox shape
        bbox_mask_full = np.zeros((h_img, w_img), dtype=np.uint8)
        cv.fillPoly(bbox_mask_full, [box_pts], 255)
        bbox_mask_crop = bbox_mask_full[y1:y2, x1:x2]
        roi_bin = cv.bitwise_and(roi_crop, bbox_mask_crop)

        # Find largest external contour in the crop
        cnts, _ = cv.findContours(roi_bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        if not cnts:
            continue

        # Pick the largest contour by area
        largest_cnt = max(cnts, key=cv.contourArea)
        if cv.contourArea(largest_cnt) < min_area:
            continue

        # RDP simplification
        simplified = rdp_simplify(largest_cnt, epsilon=eps)
        N = len(simplified)

        if N < min_pts:
            print(
                f'[concave_points] outlier #{bbox_idx}: contour too small after '
                f'RDP ({N} pts < {min_pts}) — skipping.'
            )
            continue

        # k-curvature (closed contour)
        kappa = compute_k_curvature(simplified, k)

        # ROI detection
        t0   = float(np.mean(kappa))
        rois = _find_rois_recursive(kappa, t0, lmin, lmax, delta_t, k)

        # Full-image version of the simplified contour (for concavity test)
        full_simplified = simplified + np.array([[x1, y1]], dtype=np.float64)

        # POI selection + concavity test
        for roi in rois:
            poi_idx = _select_point_of_interest(roi, kappa)

            # Metadata
            poi_kappa = float(kappa[poi_idx])

            roi_start, roi_end = roi
            roi_len = (roi_end - roi_start) % N + 1
            roi_kappas = [kappa[(roi_start + j) % N] for j in range(roi_len)]
            roi_kappa_mean = float(np.mean(roi_kappas))
            roi_kappa_max  = float(np.max(roi_kappas))

            # Chord length between ±k neighbors
            p_prev = full_simplified[(poi_idx - k) % N]
            p_next = full_simplified[(poi_idx + k) % N]
            chord_len = float(np.linalg.norm(p_next - p_prev))

            # Map to full-image coords
            lx = simplified[poi_idx][0]
            ly = simplified[poi_idx][1]
            gx = int(round(lx)) + x1
            gy = int(round(ly)) + y1
            gx = max(0, min(w_img - 1, gx))
            gy = max(0, min(h_img - 1, gy))

            # Concavity test
            if _is_concave(full_simplified, poi_idx, seg_binary, k):
                result[bbox_idx].append({
                    'x':              gx,
                    'y':              gy,
                    'kappa':          poi_kappa,
                    'roi_len':        roi_len,
                    'roi_kappa_mean': roi_kappa_mean,
                    'roi_kappa_max':  roi_kappa_max,
                    'chord_len':      chord_len,
                })

    return result
