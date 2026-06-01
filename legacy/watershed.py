# watershed.py
# Optional watershed split of fused instrument blobs.
#
# Call watershed_split() AFTER segment_instruments() whenever the outlier
# analysis reports fused (touching) instruments.  The function runs a
# marker-based watershed on the distance transform of the binary mask and
# returns the cut line (boundary_mask), the split bboxes, and the distance
# map — all as plain NumPy arrays with no visualisation side-effects.
#
# Architecture note: this module does NOT import matplotlib.  All plotting
# lives in visualize.py.

import numpy as np
import cv2 as cv

from config import PreprocessConfig
from segmentation import _filter_by_median_area


def watershed_split(
    seg_binary: np.ndarray,
    outlier_bboxes: list[tuple],
    cfg: PreprocessConfig,
) -> tuple[list[tuple], np.ndarray, np.ndarray]:
    """Split fused instrument blobs using marker-based watershed.

    WHY: When two instruments touch, their binary blobs merge into a single
    connected region with one large bounding box.  The watershed algorithm
    treats the distance transform of the binary mask as a topographic surface:
    peaks (pixels far from the background) lie on the instrument spines and
    become seeds; the saddle point between two peaks corresponds to the contact
    zone and is where the watershed floods last, producing a clean 1-pixel cut
    line that separates the two instruments.

    Per-outlier pipeline:
      1. Crop binary mask to the oriented bbox region (no convex hull — the bbox
         mask is simpler and works reliably for straight instruments).
      2. Distance transform → definite-foreground seeds (sure_fg) at pixels
         above ws_sure_fg_threshold × local dist_max.
      3. Close sure_fg with a small kernel to bridge ridge fragments from the
         SAME instrument (serration valleys, reflection gaps).
      4. Dilate binary → definite-background (sure_bg).
         unknown = sure_bg − sure_fg (the contested contact zone).
      5. Connected components on sure_fg → integer markers.
         Background = 1.  Unknown = 0 (watershed fills these).
      6. cv.watershed on the inverted distance transform (instrument centres
         become valleys; contact saddle becomes the flood peak → boundary).
      7. Extract boundary directly from markers == −1.
         This avoids the lossy RGBA → threshold round-trip of the old code.
      8. Subtract boundary from binary crop → find new contours → oriented bboxes.

    Args:
        seg_binary:     uint8 (H, W) binary mask; instruments = 255.
        outlier_bboxes: List of (center, size, angle, area) tuples — the bboxes
                        flagged as fused by _analyze_bbox_outliers.
                        Typically: [e['bbox'] for e in outlier_analysis['outliers']]
        cfg:            PreprocessConfig; reads all ws_* and seg_* parameters.

    Returns:
        split_bboxes:  List of (center, size, angle, area) — one 4-tuple per
                       separated instrument found inside the outlier regions
                       after cutting.  Returns [] if no valid split is found.
        boundary_mask: uint8 (H, W) — watershed boundary pixels = 255, else 0.
        dist_map:      float32 (H, W) — distance-transform values; non-zero
                       only inside the processed outlier regions.
    """
    h_img, w_img = seg_binary.shape

    if not outlier_bboxes:
        return [], np.zeros((h_img, w_img), dtype=np.uint8), \
                   np.zeros((h_img, w_img), dtype=np.float32)

    dist_map      = np.zeros((h_img, w_img), dtype=np.float32)
    boundary_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    split_bboxes  = []

    # 3×3 open kernel removes single-pixel spikes left along the cut edge.
    # Kept small on purpose — a larger kernel would erode thin instrument tips.
    open_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

    # Kernel to bridge nearby seed fragments from the SAME instrument.
    # Must stay smaller than the inter-instrument gap (~50–200 px) so that
    # real contacts are NOT bridged (which would prevent the split).
    seed_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, cfg.ws_seed_merge_kernel)

    # Background dilation kernel: pushes sure_bg outward so the uncertain band
    # (unknown) reliably covers the full contact zone.
    bg_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, cfg.ws_sure_bg_dilate_kernel)

    for bbox in outlier_bboxes:
        center, size, angle = bbox[0], bbox[1], bbox[2]
        box_pts = np.int32(cv.boxPoints((center, size, angle)))

        # Axis-aligned bounding rectangle of the oriented bbox
        ax, ay, aw, ah = cv.boundingRect(box_pts)
        x1, y1 = max(0, ax),           max(0, ay)
        x2, y2 = min(w_img, ax + aw),  min(h_img, ay + ah)
        if x2 <= x1 or y2 <= y1:
            continue

        # Local oriented-bbox mask: pixels of neighbouring instruments that
        # happen to fall inside the axis-aligned crop rectangle but outside
        # the actual rotated box are masked out here.
        bbox_mask_full = np.zeros((h_img, w_img), dtype=np.uint8)
        cv.fillPoly(bbox_mask_full, [box_pts], 255)
        bbox_mask_crop = bbox_mask_full[y1:y2, x1:x2]

        # ROI = the fused-instrument blob clipped to the oriented bbox
        roi_bin = cv.bitwise_and(seg_binary[y1:y2, x1:x2], bbox_mask_crop)

        # Step 2 — Distance transform.
        # dist[r, c] = distance from pixel (r, c) to the nearest background pixel.
        # Peaks are located on the instrument spines, far from any edge.
        dist = cv.distanceTransform(roi_bin, cv.DIST_L2, cfg.ws_dist_mask_size)
        if dist.max() < 1.0:
            # Degenerate crop (empty or single-pixel blob) — nothing to split.
            continue

        dist_map[y1:y2, x1:x2] = np.maximum(dist_map[y1:y2, x1:x2], dist)

        # Step 3 — Definite foreground seeds: pixels whose distance is at least
        # ws_sure_fg_threshold × the local maximum.  These are the instrument
        # centres — far from any background, so definitely inside one instrument.
        _, sure_fg = cv.threshold(
            dist,
            cfg.ws_sure_fg_threshold * dist.max(),
            255,
            cv.THRESH_BINARY,
        )
        sure_fg = sure_fg.astype(np.uint8)

        # Close sure_fg to merge ridge fragments from the same instrument.
        # Without this, reflection gaps or serration valleys fragment the distance
        # ridge → each fragment becomes a separate seed → over-segmentation
        # (one instrument split into two or more pieces).
        sure_fg = cv.morphologyEx(sure_fg, cv.MORPH_CLOSE, seed_kernel)

        # Step 4 — Definite background and unknown zone.
        # sure_bg is wider than the instrument; unknown is the ring between them
        # that contains the contact zone.
        sure_bg = cv.dilate(roi_bin, bg_kernel, iterations=cfg.ws_sure_bg_dilate_iters)
        unknown = cv.subtract(sure_bg, sure_fg)

        # Step 5 — Integer markers for cv.watershed.
        # connectedComponents labels each sure_fg blob starting from 0;
        # +1 shifts everything so sure_fg labels ≥ 2, background = 1,
        # unknown = 0 (these are the pixels watershed will flood and label).
        _, markers = cv.connectedComponents(sure_fg)
        markers = (markers + 1).astype(np.int32)
        markers[unknown == 255] = 0

        # Step 6 — Watershed on the inverted distance transform.
        # Inversion flips the landscape: instrument centres become valleys (easy
        # to flood from seed), and the contact saddle becomes a local peak where
        # water from two basins meets → watershed boundary.
        dist_u8  = cv.normalize(dist, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        inv_dist = 255 - dist_u8
        roi_3ch  = cv.merge([inv_dist, inv_dist, inv_dist])
        cv.watershed(roi_3ch, markers)

        # Step 7 — Boundary mask directly from markers.
        # cv.watershed marks every boundary pixel with −1.  Extracting this
        # directly is exact and lossless — no RGBA colour round-trip needed.
        boundary_crop = (markers == -1).astype(np.uint8) * 255
        boundary_mask[y1:y2, x1:x2] = np.maximum(
            boundary_mask[y1:y2, x1:x2], boundary_crop
        )

        # Step 8 — Cut the binary blob along the boundary and re-detect.
        cut_crop = cv.subtract(roi_bin, boundary_crop)
        cut_crop = cv.morphologyEx(cut_crop, cv.MORPH_OPEN, open_kernel)

        cnts, _ = cv.findContours(
            cut_crop, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        for cnt in cnts:
            area = cv.contourArea(cnt)
            if area < cfg.seg_min_contour_area:
                continue
            # Shift contour coordinates from crop-local to full-image space
            # before fitting the oriented bbox.
            cnt_global = (cnt + np.array([[[x1, y1]]])).astype(np.int32)
            c, s, a = cv.minAreaRect(cnt_global)
            split_bboxes.append((c, s, a, int(area)))

    # Apply the same median-area filter used on the main bbox list so that
    # tiny post-split fragments (cut artefacts, noise specks) are discarded.
    if split_bboxes:
        cut_mask_global = cv.subtract(seg_binary, boundary_mask)
        split_bboxes = _filter_by_median_area(
            split_bboxes, cut_mask_global, cfg.seg_median_area_threshold
        )

    return split_bboxes, boundary_mask, dist_map
