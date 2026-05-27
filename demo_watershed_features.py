# demo_watershed_features.py
#
# Runs the full segmentation + watershed pipeline, then studies two kinds of
# spatial features inside each outlier blob to explore contact-detection cues.
#
# Per outlier blob two 3-panel figures are produced:
#
#   ── Figure A: Gradient analysis (|∇dist|) ────────────────────────────────
#   Panel A0 — distance-transform height map + topographic isolines  (reference landscape)
#   Panel A1 — gradient magnitude |∇dist| with top-4 lowest (yellow) and top-4 highest (red) marked
#   Panel A2 — distance map + cyan arrows from every low-gradient point to every high-gradient point
#              (each arrow labelled with the Euclidean distance in px)
#   Panel A3 — direction field of the ORIGINAL IMAGE gradient (∇img_gray): unit-length quiver
#              arrows sampled on a regular grid, coloured by |∇img| (plasma), overlaid on the
#              grayscale image — shows where actual intensity edges point across the instrument
#
#   ── Figure B: Distance-transform extrema ─────────────────────────────────
#   Panel B0 — distance-transform height map + topographic isolines  (reference landscape)
#   Panel B1 — distance-transform heatmap with top-4 nearest-to-bg (yellow) and
#              top-4 farthest-from-bg / instrument-centre (red) points marked
#   Panel B2 — distance map + cyan arrows from every nearest-to-bg point to every
#              farthest-from-bg point (each arrow labelled with the Euclidean distance in px)
#
# Usage:
#   python demo_watershed_features.py
#   IMAGE_PATH = None   → random image from ./Trays
#   IMAGE_PATH = "path" → specific file

import os
import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from config     import PreprocessConfig
from preprocess import (get_ROI_from_color, binarize_image, get_tray_crop,
                        remove_blue_background, detect_specular_reflections)
from segmentation import (
    _connect_edges, _binarize_tray,
    _find_bboxes, _filter_by_median_area, _analyze_bbox_outliers,
    apply_watershed_to_outliers,
)


# ------------------------------------------------------------------ #
#  Shared sparse-extrema helper                                       #
# ------------------------------------------------------------------ #

def _find_sparse_extrema(values, mask, n=4, min_dist=30, find_min=True):
    """Return up to n (row, col) points that are extrema of *values* inside *mask*,
    with at least min_dist pixels between any two selected points.

    Greedy: iterates candidates in sorted order (ascending for minima, descending
    for maxima) and keeps a point only if it is far enough from all already-kept
    points.

    Args:
        values:   float64 (H, W) — scalar map to search (grad_mag, dist, …).
        mask:     uint8   (H, W) — foreground mask; only pixels > 0 are considered.
        n:        maximum number of points to return.
        min_dist: minimum Euclidean separation between selected points (px).
        find_min: True → find minima; False → find maxima.

    Returns:
        List of (row, col) int tuples, length ≤ n.
    """
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return []

    vals  = values[ys, xs]
    order = np.argsort(vals) if find_min else np.argsort(vals)[::-1]

    selected = []
    for i in order:
        r, c = int(ys[i]), int(xs[i])
        if all(np.hypot(r - pr, c - pc) >= min_dist for pr, pc in selected):
            selected.append((r, c))
        if len(selected) >= n:
            break
    return selected


# ------------------------------------------------------------------ #
#  Figure A — Gradient of the distance transform                      #
# ------------------------------------------------------------------ #

def _gradient_figure(roi_bin, dist, blob_idx, cfg, img_gray=None):
    """Draw a 4-panel figure for one outlier blob (gradient-magnitude analysis).

    Panel 0 — distance transform + contour isolines (reference landscape).
    Panel 1 — |∇dist| heatmap with low (yellow circle) / high (red square) markers.
    Panel 2 — distance transform + cyan arrows from every low-gradient point to
              every high-gradient point, each labelled with the pixel distance.
    Panel 3 — direction field of the ORIGINAL IMAGE gradient (∇img_gray): unit-length
              quiver arrows sampled on a regular grid, coloured by |∇img_gray| (plasma),
              overlaid on the grayscale image crop.  If img_gray is None the panel is
              skipped.

    Args:
        roi_bin:   uint8 (H, W) — hull-masked binary ROI.
        dist:      float32 (H, W) — distance transform crop (from dist_map).
        blob_idx:  0-based index of the outlier blob (for the figure title).
        cfg:       PreprocessConfig.
        img_gray:  uint8 (H, W) — grayscale image crop, same spatial extent as dist.
                   Used only for Panel 3.  Pass None to omit that panel.
    """
    # Gradient of the distance transform (restricted to foreground)
    gy, gx   = np.gradient(dist.astype(np.float64))
    grad_mag = np.hypot(gx, gy)

    # Only search within actual foreground pixels (dist > 0)
    fg_mask = (dist > 0).astype(np.uint8)

    # Adaptive minimum separation: 10 % of the shorter crop axis, at least 15 px
    min_sep = max(int(0.10 * min(dist.shape)), 15)

    low_pts  = _find_sparse_extrema(grad_mag, fg_mask, n=4, min_dist=min_sep, find_min=True)
    high_pts = _find_sparse_extrema(grad_mag, fg_mask, n=4, min_dist=min_sep, find_min=False)

    cmap_dist = cfg.ws_heatmap_colormap

    fig, axs = plt.subplots(1, 4, figsize=(25, 6))
    fig.suptitle(
        f"[A] Gradient analysis — blob #{blob_idx + 1}  "
        f"({dist.shape[1]}×{dist.shape[0]} px crop)",
        fontsize=11, fontweight="bold",
    )

    # ── Panel 0: distance transform + topographic isolines ──────────
    im0 = axs[0].imshow(dist, cmap=cmap_dist, interpolation='bilinear')
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04, label='Distance (px)')
    if dist.max() > 0:
        axs[0].contour(
            dist,
            levels=np.linspace(0, dist.max(), 12)[1:-1],
            colors='white', linewidths=0.5, alpha=0.5,
        )
    axs[0].set_title("Distance transform\n(watershed height map + isolines)")
    axs[0].axis('off')

    # ── Panel 1: gradient magnitude + marked extrema ─────────────────
    im1 = axs[1].imshow(grad_mag, cmap='viridis', interpolation='bilinear')
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04, label='|∇dist|  (px / px)')

    for r, c in low_pts:
        axs[1].plot(c, r, 'o', color='yellow', markersize=9,
                    markeredgecolor='black', markeredgewidth=1.0, zorder=5,
                    label='low ∇ (min)')
    for r, c in high_pts:
        axs[1].plot(c, r, 's', color='red', markersize=9,
                    markeredgecolor='black', markeredgewidth=1.0, zorder=5,
                    label='high ∇ (max)')

    handles, labels = axs[1].get_legend_handles_labels()
    axs[1].legend(dict(zip(labels, handles)).values(),
                  dict(zip(labels, handles)).keys(),
                  loc='lower right', fontsize=7, framealpha=0.65)
    axs[1].set_title("|∇dist| magnitude\nyellow = 4 lowest  ·  red = 4 highest")
    axs[1].axis('off')

    # ── Panel 2: arrows from each low-gradient pt to each high-gradient pt ──
    axs[2].imshow(dist, cmap=cmap_dist, interpolation='bilinear')

    for r, c in low_pts:
        axs[2].plot(c, r, 'o', color='yellow', markersize=9,
                    markeredgecolor='black', markeredgewidth=1.0, zorder=5)
    for r, c in high_pts:
        axs[2].plot(c, r, 's', color='red', markersize=9,
                    markeredgecolor='black', markeredgewidth=1.0, zorder=5)

    # All 4 × 4 = 16 pairs: arrow from low → high with distance label
    for (lr, lc) in low_pts:
        for (hr, hc) in high_pts:
            d_px  = np.hypot(hr - lr, hc - lc)
            mid_r = (lr + hr) / 2.0
            mid_c = (lc + hc) / 2.0
            axs[2].annotate(
                '',
                xy=(hc, hr), xytext=(lc, lr),
                arrowprops=dict(
                    arrowstyle='->', color='cyan',
                    lw=1.1, mutation_scale=12,
                ),
                zorder=4,
            )
            axs[2].text(
                mid_c, mid_r, f"{d_px:.0f}",
                color='white', fontsize=6, ha='center', va='center', zorder=6,
                bbox=dict(facecolor='black', alpha=0.45, pad=1, edgecolor='none'),
            )

    axs[2].set_title(
        "Low → High gradient arrows  (all 4×4 pairs)\n"
        "yellow = low ∇  ·  red = high ∇  ·  cyan arrow = direction  ·  label = px"
    )
    axs[2].axis('off')

    # ── Panel 3: image gradient direction field (quiver) ────────────────
    # Uses the gradient of the ORIGINAL grayscale image (not the distance
    # transform) so the arrows reflect the actual intensity structure of the
    # instrument — edges, reflections, surface texture — rather than the
    # synthetic watershed landscape.
    if img_gray is not None:
        img_f = img_gray.astype(np.float64)
        igy, igx = np.gradient(img_f)
        img_grad_mag = np.hypot(igx, igy)

        # Show the grayscale image as background so the arrows are interpretable
        axs[3].imshow(img_gray, cmap='gray', interpolation='bilinear', alpha=0.80)

        # Grid step: ~20 samples along the shorter axis, minimum 5 px apart
        step = max(5, min(dist.shape) // 20)
        rows_g = np.arange(step // 2, dist.shape[0], step)
        cols_g = np.arange(step // 2, dist.shape[1], step)
        C_g, R_g = np.meshgrid(cols_g, rows_g)

        igx_s = igx[R_g, C_g]
        igy_s = igy[R_g, C_g]
        fg_s  = fg_mask[R_g, C_g]
        nm_s  = np.hypot(igx_s, igy_s)

        # Keep only foreground pixels with a non-trivial gradient
        valid = (fg_s > 0) & (nm_s > 1e-3)
        ux_s = np.where(valid, igx_s / (nm_s + 1e-10), np.nan)
        # Negate vy: imshow places row 0 at top (y increases downward), so
        # a positive row-gradient (gy > 0) must point downward in display.
        uy_s = np.where(valid, -igy_s / (nm_s + 1e-10), np.nan)
        mag_s = np.where(valid, nm_s, np.nan)

        Cv, Rv = C_g[valid], R_g[valid]
        Uv, Vv = ux_s[valid], uy_s[valid]
        Mv     = mag_s[valid]

        if Cv.size > 0:
            peak_mag = img_grad_mag[fg_mask > 0].max() if fg_mask.any() else 1.0
            qv = axs[3].quiver(
                Cv, Rv, Uv, Vv,
                Mv,                          # colour by image-gradient magnitude
                cmap='plasma',
                clim=(0, peak_mag),
                angles='xy', scale_units='xy',
                scale=1.0 / step,            # one arrow = step pixels long
                width=0.003,
                headwidth=4, headlength=4,
                alpha=0.90,
                zorder=3,
            )
            fig.colorbar(qv, ax=axs[3], fraction=0.046, pad=0.04,
                         label='|∇img|  (intensity / px)')

        axs[3].set_title(
            "Image gradient direction field  (∇img_gray)\n"
            f"unit arrows every {step} px  ·  colour = |∇img|  ·  bg = grayscale image"
        )
    else:
        axs[3].axis('off')
        axs[3].set_title("Image gradient\n(img_gray not provided)")

    axs[3].axis('off')

    plt.tight_layout()


# ------------------------------------------------------------------ #
#  Figure B — Distance-transform extrema (nearest / farthest from bg) #
# ------------------------------------------------------------------ #

def _distance_figure(roi_bin, dist, blob_idx, cfg):
    """Draw a 3-panel figure for one outlier blob (distance-transform extrema).

    Finds the top-4 points with the LOWEST distance-transform value inside the
    foreground (nearest to the sure background / contact zone) and the top-4
    points with the HIGHEST distance-transform value (farthest from background /
    instrument spine or centre).  Mirrors the gradient figure but operates
    directly on the distance values instead of |∇dist|.

    Panel 0 — distance transform + contour isolines (reference landscape).
    Panel 1 — distance-transform heatmap with nearest-to-bg (yellow circles)
              and farthest-from-bg / centre (red squares) marked.
    Panel 2 — distance transform + cyan arrows from every nearest-to-bg point
              to every farthest-from-bg point, each labelled with the pixel distance.

    Args:
        roi_bin:   uint8 (H, W) — hull-masked binary ROI.
        dist:      float32 (H, W) — distance transform crop (from dist_map).
        blob_idx:  0-based index of the outlier blob (for the figure title).
        cfg:       PreprocessConfig.
    """
    dist_f64 = dist.astype(np.float64)

    # Only search within actual foreground pixels (dist > 0).
    # "nearest to bg" = lowest positive dist value (just inside the object border)
    # "farthest from bg" = highest dist value (instrument spine / centre)
    fg_mask = (dist > 0).astype(np.uint8)

    # Adaptive minimum separation: 10 % of the shorter crop axis, at least 15 px
    min_sep = max(int(0.10 * min(dist.shape)), 15)

    near_pts = _find_sparse_extrema(dist_f64, fg_mask, n=4, min_dist=min_sep, find_min=True)
    far_pts  = _find_sparse_extrema(dist_f64, fg_mask, n=4, min_dist=min_sep, find_min=False)

    cmap_dist = cfg.ws_heatmap_colormap

    fig, axs = plt.subplots(1, 3, figsize=(19, 6))
    fig.suptitle(
        f"[B] Distance-transform extrema — blob #{blob_idx + 1}  "
        f"({dist.shape[1]}×{dist.shape[0]} px crop)",
        fontsize=11, fontweight="bold",
    )

    # ── Panel 0: distance transform + topographic isolines ──────────
    im0 = axs[0].imshow(dist, cmap=cmap_dist, interpolation='bilinear')
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04, label='Distance (px)')
    if dist.max() > 0:
        axs[0].contour(
            dist,
            levels=np.linspace(0, dist.max(), 12)[1:-1],
            colors='white', linewidths=0.5, alpha=0.5,
        )
    axs[0].set_title("Distance transform\n(watershed height map + isolines)")
    axs[0].axis('off')

    # ── Panel 1: distance heatmap + marked extrema ───────────────────
    im1 = axs[1].imshow(dist, cmap=cmap_dist, interpolation='bilinear')
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04, label='Distance (px)')

    for r, c in near_pts:
        axs[1].plot(c, r, 'o', color='yellow', markersize=9,
                    markeredgecolor='black', markeredgewidth=1.0, zorder=5,
                    label='nearest to bg (min dist)')
    for r, c in far_pts:
        axs[1].plot(c, r, 's', color='red', markersize=9,
                    markeredgecolor='black', markeredgewidth=1.0, zorder=5,
                    label='farthest from bg (max dist)')

    handles, labels = axs[1].get_legend_handles_labels()
    axs[1].legend(dict(zip(labels, handles)).values(),
                  dict(zip(labels, handles)).keys(),
                  loc='lower right', fontsize=7, framealpha=0.65)
    axs[1].set_title(
        "Distance-transform extrema\n"
        "yellow = 4 nearest-to-bg  ·  red = 4 farthest-from-bg (centres)"
    )
    axs[1].axis('off')

    # ── Panel 2: arrows from each near-bg pt to each far-from-bg pt ──
    axs[2].imshow(dist, cmap=cmap_dist, interpolation='bilinear')

    for r, c in near_pts:
        axs[2].plot(c, r, 'o', color='yellow', markersize=9,
                    markeredgecolor='black', markeredgewidth=1.0, zorder=5)
    for r, c in far_pts:
        axs[2].plot(c, r, 's', color='red', markersize=9,
                    markeredgecolor='black', markeredgewidth=1.0, zorder=5)

    # All 4 × 4 = 16 pairs: arrow from near-bg → far-from-bg with distance label
    for (nr, nc) in near_pts:
        for (fr, fc) in far_pts:
            d_px  = np.hypot(fr - nr, fc - nc)
            mid_r = (nr + fr) / 2.0
            mid_c = (nc + fc) / 2.0
            axs[2].annotate(
                '',
                xy=(fc, fr), xytext=(nc, nr),
                arrowprops=dict(
                    arrowstyle='->', color='cyan',
                    lw=1.1, mutation_scale=12,
                ),
                zorder=4,
            )
            axs[2].text(
                mid_c, mid_r, f"{d_px:.0f}",
                color='white', fontsize=6, ha='center', va='center', zorder=6,
                bbox=dict(facecolor='black', alpha=0.45, pad=1, edgecolor='none'),
            )

    axs[2].set_title(
        "Near-bg → Far-from-bg arrows  (all 4×4 pairs)\n"
        "yellow = nearest-to-bg  ·  red = farthest-from-bg  ·  cyan = direction  ·  label = px"
    )
    axs[2].axis('off')

    plt.tight_layout()


# ------------------------------------------------------------------ #
#  Main demo                                                          #
# ------------------------------------------------------------------ #

def run_demo(image_path=None):
    """Run the full pipeline on one image, then show both feature-analysis
    figures (gradient + distance extrema) for every outlier blob detected.

    Args:
        image_path: path to a specific image, or None for a random image in ./Trays.
    """
    cfg = PreprocessConfig()

    # ── Image loading ────────────────────────────────────────────────
    if image_path is None:
        all_paths = [
            os.path.join(wd, f)
            for wd, _, files in os.walk("./Trays")
            for f in files
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
        ]
        if not all_paths:
            raise FileNotFoundError("No images found in ./Trays")
        image_path = all_paths[np.random.randint(0, len(all_paths))]

    print(f"[demo] {image_path}")
    img_bgr = cv.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load: {image_path}")
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    img_rgb = cv.resize(img_rgb, cfg.image_dims, interpolation=cv.INTER_AREA)

    # ── Preprocessing ────────────────────────────────────────────────
    roi_crop, _, roi_bbox = get_ROI_from_color(img_rgb, cfg)
    binary_mask           = binarize_image(roi_crop, cfg)
    tray_masked, _, _     = get_tray_crop(roi_crop, binary_mask, cfg)
    detect_specular_reflections(tray_masked, cfg)   # side-effect: inpaints in place
    tray_no_bg            = remove_blue_background(tray_masked, cfg)

    # Grayscale version of the processed image — used for the image-gradient
    # direction field in Panel A3.  Convert from RGB (tray_no_bg is RGB).
    img_gray = cv.cvtColor(tray_no_bg, cv.COLOR_RGB2GRAY)

    # ── Segmentation (mirrors segment_instruments) ───────────────────
    tray_closed      = _connect_edges(tray_no_bg, cfg)
    seg_binary       = _binarize_tray(tray_closed, cfg)
    bboxes           = _find_bboxes(seg_binary, cfg.seg_min_contour_area)
    bboxes, _        = _filter_by_median_area(bboxes, seg_binary, cfg.seg_median_area_threshold)
    outlier_analysis = _analyze_bbox_outliers(
        seg_binary, bboxes,
        component_threshold      = cfg.seg_outlier_grade_threshold,
        secondary_ratio          = cfg.seg_outlier_secondary_ratio,
        weight_area              = cfg.seg_outlier_weight_area,
        weight_edge              = cfg.seg_outlier_weight_edge,
        weight_fill              = cfg.seg_outlier_weight_fill,
        edge_magnitude_threshold = cfg.seg_edge_magnitude_threshold,
    )

    print(f"  Scenario     : {outlier_analysis['scenario']}")
    print(f"  Outlier blobs: {len(outlier_analysis['outliers'])}")

    if not outlier_analysis['outliers']:
        print("  No fused-instrument blobs found.\n"
              "  Try an image that contains two touching instruments.\n"
              "  You can lower cfg.seg_outlier_grade_threshold to force a detection.")
        return

    # ── Watershed ────────────────────────────────────────────────────
    ws_result = apply_watershed_to_outliers(seg_binary, outlier_analysis, cfg)
    if ws_result is None:
        print("  Watershed returned no result.")
        return
    dist_map, segment_overlay = ws_result

    # ── Per-blob analysis ────────────────────────────────────────────
    h_img, w_img = seg_binary.shape

    for blob_idx, entry in enumerate(outlier_analysis['outliers']):
        bbox = entry['bbox']
        center, size, angle = bbox[0], bbox[1], bbox[2]
        box_pts = np.int32(cv.boxPoints((center, size, angle)))

        full_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cv.fillPoly(full_mask, [box_pts], 255)

        ax, ay, aw, ah = cv.boundingRect(box_pts)
        x1, y1 = max(0, ax),           max(0, ay)
        x2, y2 = min(w_img, ax + aw),  min(h_img, ay + ah)
        if x2 <= x1 or y2 <= y1:
            continue

        # Re-derive hull mask (identical logic to apply_watershed_to_outliers)
        seg_clip = cv.bitwise_and(
            seg_binary[y1:y2, x1:x2], full_mask[y1:y2, x1:x2]
        )
        cnts, _ = cv.findContours(seg_clip, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        main_cnt  = max(cnts, key=cv.contourArea)
        hull      = cv.convexHull(main_cnt)
        hull_mask = np.zeros(seg_clip.shape, dtype=np.uint8)
        cv.fillPoly(hull_mask, [hull], 255)
        roi_bin   = cv.bitwise_and(seg_binary[y1:y2, x1:x2], hull_mask)

        dist_crop = dist_map[y1:y2, x1:x2]
        if dist_crop.max() < 1.0:
            print(f"  Blob #{blob_idx + 1}: distance transform is empty, skipping.")
            continue

        print(f"  Blob #{blob_idx + 1}: crop {dist_crop.shape[1]}×{dist_crop.shape[0]} px  "
              f"dist_max={dist_crop.max():.1f}")

        # Grayscale image crop aligned with the blob bbox
        gray_crop = img_gray[y1:y2, x1:x2]

        # Figure A — gradient of the distance transform + image gradient direction
        _gradient_figure(roi_bin, dist_crop, blob_idx, cfg, img_gray=gray_crop)

        # Figure B — distance-transform extrema (nearest / farthest from background)
        _distance_figure(roi_bin, dist_crop, blob_idx, cfg)

    plt.show()
    plt.close('all')


if __name__ == "__main__":
    IMAGE_PATH = "./Trays/IMG_3353.jpg"   # None = random image from ./Trays; or set a specific path
    run_demo(image_path=IMAGE_PATH)
