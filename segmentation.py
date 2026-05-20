# segmentation.py
# Segments surgical instruments from the background-removed tray image.
# Pipeline: binarize (Sauvola) -> contour detection -> bounding boxes -> visualise

import io
import os
import platform
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

if platform.system() == "Linux":
    os.environ["QT_QPA_PLATFORM"] = "xcb" # Force x11 on wayland DEs
    os.environ["DISPLAY"] = os.environ.get("DISPLAY", ":0")

from skimage.filters import threshold_sauvola

from config     import PreprocessConfig
from preprocess import (get_ROI_from_color, binarize_image, get_tray_crop,
                        remove_blue_background, detect_specular_reflections)
from utils      import edge_Laplace

# ------------------------------------------------------------------ #
#  Step 0 — Morphological close to connect nearby edges              #
# ------------------------------------------------------------------ #

def _connect_edges(tray_no_bg, cfg):
    """Apply a morphological close on each RGB channel to bridge small
    gaps between instrument edges before binarization."""
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, cfg.seg_close_kernel_dims)
    return cv.morphologyEx(tray_no_bg, cv.MORPH_CLOSE, kernel)


# ------------------------------------------------------------------ #
#  Step 1 — Sauvola binarization                                     #
# ------------------------------------------------------------------ #

def _binarize_tray(tray_no_bg, cfg):
    """Sauvola local threshold on the background-removed image, followed by
    a morphological close to fill small holes in the binary mask.
    Sauvola adapts the threshold per-pixel based on local mean and std,
    preserving fine instrument detail better than a global Otsu threshold. Dilation afterwards."""
    gray   = cv.cvtColor(tray_no_bg, cv.COLOR_RGB2GRAY)
    thresh = threshold_sauvola(gray, window_size=cfg.sauvola_window_size, k=cfg.sauvola_k)
    binary = (gray > thresh).astype(np.uint8) * 255
    # dilation and closing
    dilating_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, cfg.seg_close_kernel_dims)
    dilated_img = cv.morphologyEx(binary, cv.MORPH_DILATE, dilating_kernel)
    opening_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, cfg.open_kernel_dims)
    opened_img = cv.morphologyEx(dilated_img, cv.MORPH_OPEN, opening_kernel) 
    return opened_img

# ------------------------------------------------------------------ #
#  Step 3 — Contour detection and bounding boxes                     #
# ------------------------------------------------------------------ #

def _find_bboxes(binary, min_area):
    """Return a list of minimum-area oriented bounding boxes for each external
    contour whose area exceeds min_area.

    Returns:
        list of (center, size, angle, contour_area) where center=(cx,cy),
        size=(w,h), angle is the rotation in degrees.
    """
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < min_area:
            continue
        center, size, angle = cv.minAreaRect(cnt)
        bboxes.append((center, size, angle, int(area)))
    # Largest first
    bboxes.sort(key=lambda b: b[3], reverse=True)
    return bboxes


# ------------------------------------------------------------------ #
#  Step 4 — Median-area outlier filter                               #
# ------------------------------------------------------------------ #

def _filter_by_median_area(bboxes, binary, threshold):
    """Discard bounding boxes below `threshold` × median of the effective bbox area.

    The filter metric is  positive_pixels / fill_ratio  which simplifies to
    the oriented-bbox area (w × h).  This is deliberately different from the
    contour polygon area stored in b[3]: a thin diagonal instrument has a small
    polygon area but a large oriented-bbox footprint, and we want to keep it.

    Args:
        bboxes:    list of (center, size, angle, contour_area).
        binary:    uint8 binary mask used to count positive pixels per bbox.
        threshold: fraction of the median metric used as the lower bound.

    Returns:
        Filtered list, same format, same ordering.
    """
    if not bboxes:
        return bboxes

    def _effective_area(bbox):
        stats = _compute_bbox_stats(binary, bbox)
        fill_ratio = stats['fill_ratio']
        if fill_ratio <= 0:
            return 0.0
        return stats['positive_pixels'] / fill_ratio  # == bbox_area (w × h)

    metrics     = [_effective_area(b) for b in bboxes]
    median_val  = float(np.median(metrics))
    min_val     = threshold * median_val
    per_bbox    = {id(b): m for b, m in zip(bboxes, metrics)}
    filtered    = [b for b, m in zip(bboxes, metrics) if m >= min_val]
    filter_stats = {
        'median':    median_val,
        'threshold': threshold,
        'cutoff':    min_val,
        'per_bbox':  per_bbox,
    }
    return filtered, filter_stats


# ------------------------------------------------------------------ #
#  Step 5 — Outlier detection (median-area filter)                   #
# ------------------------------------------------------------------ #

def _compute_bbox_stats(binary, bbox):
    """Compute per-bbox statistics used for outlier detection and future SVM features.

    Returns a dict with:
      contour_area    – area of the contour polygon (px²)
      bbox_area       – oriented bounding-box area  w × h  (px²)
      perimeter       – oriented bounding-box perimeter  2*(w+h)  (px)
      width, height   – sides of the oriented bbox  (px)
      fill_ratio      – contour_area / bbox_area  (0–1)
      positive_pixels – white pixels inside the oriented bbox from binary mask
    """
    center, size, angle, contour_area = bbox
    w, h = size
    bbox_area = float(w * h)

    box_pts  = np.int32(cv.boxPoints(((center[0], center[1]), size, angle)))
    roi_mask = np.zeros(binary.shape[:2], dtype=np.uint8)
    cv.fillPoly(roi_mask, [box_pts], 255)
    positive_pixels = int(np.count_nonzero(cv.bitwise_and(binary, roi_mask)))

    return {
        'contour_area':    contour_area,
        'bbox_area':       bbox_area,
        'perimeter':       float(2 * (w + h)),
        'width':           float(w),
        'height':          float(h),
        'fill_ratio':      contour_area / bbox_area if bbox_area > 0 else 0.0,
        'positive_pixels': positive_pixels,
    }


def _edge_length_in_bbox(edge_binary, bbox):
    """Count edge pixels inside an oriented bounding box."""
    img_h, img_w = edge_binary.shape
    center, size, angle = bbox[0], bbox[1], bbox[2]
    mask    = np.zeros((img_h, img_w), dtype=np.uint8)
    box_pts = np.int32(cv.boxPoints((center, size, angle)))
    cv.fillPoly(mask, [box_pts], 255)
    return int(np.count_nonzero(edge_binary[mask > 0]))


def _analyze_bbox_outliers(binary, bboxes, component_threshold=1.5,
                           secondary_ratio=1.5):
    """Classify bboxes into normal and outlier groups using a grade median filter
    with a secondary consistency check for the multiple-outlier scenario.

    Each component is scaled by the minimum value seen across all bboxes (→ [1, ∞)):

        area_score = bbox_area        / min_bbox_area
        edge_score = edge_length      / min_edge_length
        fill_score = (1/fill_ratio)   / min_inv_fill_ratio

    Primary pass: flag any bbox whose grade exceeds component_threshold × median_grade,
    where grade = 0.50·area_score + 0.25·edge_score + 0.25·fill_score.

    Secondary pass (applied when 2+ candidates survive the primary pass):
    among the flagged candidates keep only those whose grade is within
    secondary_ratio of the maximum candidate grade:

        keep if  grade ≥ max_candidate_grade / secondary_ratio

    This ensures "multiple" only fires when the outlier grades are genuinely
    close to each other. A candidate that is far below the true peak (e.g.
    grade 4.93 vs peak 11.71 with ratio 2.0 → cutoff 5.86) is reclassified
    as normal, collapsing the scenario back to 'single'.

    Scenarios
    ---------
    'none'     – no bbox exceeds the grade cutoff.
    'single'   – exactly one bbox survives both passes; likely two touching instruments fused.
    'multiple' – two or more bboxes survive both passes, with similar grades.

    Returns
    -------
    dict with keys:
      'scenario'          : 'none' | 'single' | 'multiple'
      'outliers'          : list of {'bbox': ..., 'area_score', 'edge_score', 'fill_score', 'grade', **stats}
      'normal'            : same structure
      'median_area_score' : float
      'median_edge_score' : float
      'median_fill_score' : float
      'median_grade'      : float
      'grade_cutoff'      : float – component_threshold × median_grade
    """
    if not bboxes:
        return {
            'scenario': 'none', 'outliers': [], 'normal': [],
            'median_area_score': 0.0, 'median_edge_score': 0.0,
            'median_fill_score': 0.0, 'median_grade': 0.0,
            'grade_cutoff': 0.0,
            'primary_candidates': [], 'max_candidate_grade': 0.0,
            'secondary_cutoff': 0.0, 'secondary_ratio': secondary_ratio,
        }

    edge_img    = edge_Laplace(binary.astype(np.float32))
    edge_binary = (np.abs(edge_img) > 1.0).astype(np.uint8) * 255

    bbox_areas   = [b[1][0] * b[1][1] for b in bboxes]
    edge_lengths = [_edge_length_in_bbox(edge_binary, b) for b in bboxes]
    inv_fills    = [(b[1][0] * b[1][1]) / b[3] if b[3] > 0 else 0.0
                    for b in bboxes]

    min_bbox_area = max(float(np.min(bbox_areas)),  1.0)
    min_edge_len  = max(float(np.min(edge_lengths)), 1.0)
    min_inv_fill  = max(float(np.min(inv_fills)),    1e-6)

    area_scores = [ba  / min_bbox_area for ba  in bbox_areas]
    edge_scores = [el  / min_edge_len  for el  in edge_lengths]
    fill_scores = [ivf / min_inv_fill  for ivf in inv_fills]
    grades      = [0.50 * a + 0.25 * e + 0.25 * f
                   for a, e, f in zip(area_scores, edge_scores, fill_scores)]

    median_area_score = float(np.median(area_scores))
    median_edge_score = float(np.median(edge_scores))
    median_fill_score = float(np.median(fill_scores))
    median_grade      = float(np.median(grades))

    grade_cutoff = component_threshold * median_grade

    outliers, normal = [], []
    for bbox, a_sc, e_sc, f_sc, grade in zip(
            bboxes, area_scores, edge_scores, fill_scores, grades):
        stats = _compute_bbox_stats(binary, bbox)
        entry = {'bbox': bbox,
                 'area_score': a_sc, 'edge_score': e_sc, 'fill_score': f_sc,
                 'grade': grade, **stats}
        if grade > grade_cutoff:
            outliers.append(entry)
        else:
            normal.append(entry)

    # Snapshot primary candidates before secondary pass
    primary_candidates   = list(outliers)
    max_candidate_grade  = max((e['grade'] for e in outliers), default=0.0)
    secondary_cutoff_val = 0.0

    # Secondary pass: among candidates, drop those far below the peak grade
    if len(outliers) >= 2:
        secondary_cutoff_val = max_candidate_grade / secondary_ratio
        true_outliers = [e for e in outliers if e['grade'] >= secondary_cutoff_val]
        normal       += [e for e in outliers if e['grade'] <  secondary_cutoff_val]
        outliers      = true_outliers

    if   len(outliers) == 0: scenario = 'none'
    elif len(outliers) == 1: scenario = 'single'
    else:                    scenario = 'multiple'

    return {
        'scenario':            scenario,
        'outliers':            outliers,
        'normal':              normal,
        'median_area_score':   median_area_score,
        'median_edge_score':   median_edge_score,
        'median_fill_score':   median_fill_score,
        'median_grade':        median_grade,
        'grade_cutoff':        grade_cutoff,
        'primary_candidates':  primary_candidates,
        'max_candidate_grade': max_candidate_grade,
        'secondary_cutoff':    secondary_cutoff_val,
        'secondary_ratio':     secondary_ratio,
    }


# ------------------------------------------------------------------ #
#  Step 6 — Watershed on outlier bboxes (debug / trial)             #
# ------------------------------------------------------------------ #

def apply_watershed_to_outliers(seg_binary, outlier_analysis):
    """Apply watershed localised to each outlier bbox to split fused instruments.

    Uses the distance transform of the binary mask as the flooding height map.
    Local maxima (instrument centres) become seeds; the contact-line valley
    becomes the watershed boundary.

    Args:
        seg_binary:       uint8 binary mask (H, W).
        outlier_analysis: dict from _analyze_bbox_outliers.

    Returns:
        RGB uint8 image (H, W, 3) with each recovered segment coloured
        distinctly and boundaries in red, or None if there are no outliers.
    """
    if not outlier_analysis or not outlier_analysis['outliers']:
        return None

    h_img, w_img = seg_binary.shape
    result = cv.cvtColor(seg_binary, cv.COLOR_GRAY2RGB)

    seg_colors = [
        ( 50, 200,  50),
        ( 50,  50, 220),
        (220, 200,  50),
        (220,  50, 220),
        ( 50, 220, 220),
    ]

    for entry in outlier_analysis['outliers']:
        bbox = entry['bbox']
        center, size, angle = bbox[0], bbox[1], bbox[2]
        box_pts = np.int32(cv.boxPoints((center, size, angle)))

        full_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cv.fillPoly(full_mask, [box_pts], 255)

        ax, ay, aw, ah = cv.boundingRect(box_pts)
        x1, y1 = max(0, ax),         max(0, ay)
        x2, y2 = min(w_img, ax + aw), min(h_img, ay + ah)
        if x2 <= x1 or y2 <= y1:
            continue

        roi_bin = cv.bitwise_and(seg_binary[y1:y2, x1:x2],
                                  full_mask[y1:y2, x1:x2])

        dist = cv.distanceTransform(roi_bin, cv.DIST_L2, 5)
        if dist.max() < 1.0:
            continue

        _, sure_fg = cv.threshold(dist, 0.5 * dist.max(), 255, cv.THRESH_BINARY)
        sure_fg = sure_fg.astype(np.uint8)

        kernel  = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        sure_bg = cv.dilate(roi_bin, kernel, iterations=2)
        unknown = cv.subtract(sure_bg, sure_fg)

        num_labels, markers = cv.connectedComponents(sure_fg)
        markers = (markers + 1).astype(np.int32)
        markers[unknown == 255] = 0

        # Invert distance transform: centres become valleys, contact saddle
        # becomes the local peak where watershed places the boundary.
        dist_u8  = cv.normalize(dist, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        inv_dist = 255 - dist_u8
        roi_3ch  = cv.merge([inv_dist, inv_dist, inv_dist])
        cv.watershed(roi_3ch, markers)

        for label in range(2, num_labels + 1):
            color = seg_colors[(label - 2) % len(seg_colors)]
            result[y1:y2, x1:x2][markers == label] = color

        result[y1:y2, x1:x2][markers == -1] = (255, 50, 50)

    return result


# ------------------------------------------------------------------ #
#  Public API                                                         #
# ------------------------------------------------------------------ #

def segment_instruments(tray_no_bg, cfg):
    """Detect instrument bounding boxes from the background-removed tray image.

    Args:
        tray_no_bg: RGB image from remove_blue_background (H, W, 3);
                    blue background and outer border are zeroed.
        cfg:        PreprocessConfig.

    Returns:
        seg_binary: uint8 binary mask (H, W); instruments = 255.
        bboxes:     list of (x, y, w, h, area), sorted largest-first.
    """
    tray_closed      = _connect_edges(tray_no_bg, cfg)
    seg_binary       = _binarize_tray(tray_closed, cfg)
    bboxes           = _find_bboxes(seg_binary, cfg.seg_min_contour_area)
    bboxes, filter_stats = _filter_by_median_area(bboxes, seg_binary, cfg.seg_median_area_threshold)
    outlier_analysis     = _analyze_bbox_outliers(
        seg_binary, bboxes,
        cfg.seg_outlier_grade_threshold,
        cfg.seg_outlier_secondary_ratio,
    )
    return seg_binary, bboxes, outlier_analysis, filter_stats


# ------------------------------------------------------------------ #
#  Visualisation                                                      #
# ------------------------------------------------------------------ #

def visualise_results(tray_masked, tray_no_bg, seg_binary, bboxes,
                      reflection_mask=None, image_label=None, outlier_analysis=None,
                      filter_stats=None, img_rgb=None, roi_bbox=None,
                      watershed_img=None):
    """Plot 6-panel grid (2×3):
      [0] Original image + ROI bbox  [1] Specular reflections  [2] Binary mask
      [3] Edges + bboxes             [4] Watershed on outliers [5] Stats table
    """
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    if image_label:
        fig.suptitle(image_label, fontsize=13, fontweight="bold")

    outlier_bboxes = set()
    if outlier_analysis and outlier_analysis['outliers']:
        outlier_bboxes = {id(e['bbox']) for e in outlier_analysis['outliers']}

    # ── Panel 0: original image with ROI bounding box ────────────────
    if img_rgb is not None and roi_bbox is not None:
        roi_viz = img_rgb.copy()
        x0, y0, rw, rh = roi_bbox
        cv.rectangle(roi_viz, (x0, y0), (x0 + rw, y0 + rh), (0, 255, 0), thickness=4)
        axs.flat[0].imshow(roi_viz)
        axs.flat[0].set_title("Original image + ROI")
    else:
        axs.flat[0].imshow(tray_masked)
        axs.flat[0].set_title("Tray masked")
    axs.flat[0].axis("off")

    # ── Panel 1: specular reflections ────────────────────────────────
    if reflection_mask is not None:
        axs.flat[1].imshow(reflection_mask, cmap="gray")
        axs.flat[1].set_title("Specular reflections removed")
    else:
        axs.flat[1].imshow(tray_no_bg)
        axs.flat[1].set_title("Background removed (H only)")
    axs.flat[1].axis("off")

    # ── Panel 2: Sauvola binary mask ─────────────────────────────────
    axs.flat[2].imshow(seg_binary, cmap="gray")
    axs.flat[2].set_title("Sauvola binary mask")
    axs.flat[2].axis("off")

    # ── Panel 3: Laplacian edges + oriented bboxes ───────────────────
    edge_img    = edge_Laplace(seg_binary.astype(np.float32))
    edge_binary = np.abs(edge_img) > 1.0
    overlay     = cv.cvtColor(seg_binary, cv.COLOR_GRAY2RGB)
    h_img, w_img = seg_binary.shape
    for bbox in bboxes:
        center, size, angle = bbox[0], bbox[1], bbox[2]
        bbox_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cv.fillPoly(bbox_mask, [np.int32(cv.boxPoints((center, size, angle)))], 255)
        colour_rgb = (255, 0, 0) if id(bbox) in outlier_bboxes else (0, 255, 0)
        overlay[edge_binary & (bbox_mask > 0)] = colour_rgb

    bbox_ax = axs.flat[3]
    bbox_ax.imshow(overlay)
    bbox_ax.set_title(f"Laplacian edges + bboxes ({len(bboxes)} instruments)")
    bbox_ax.axis("off")

    for i, bbox in enumerate(bboxes):
        center, size, angle, area = bbox
        is_outlier = id(bbox) in outlier_bboxes
        colour = "red" if is_outlier else "lime"
        box_points = np.int32(cv.boxPoints(((center[0], center[1]), size, angle)))
        bbox_ax.add_patch(mpatches.Polygon(
            box_points, closed=True,
            linewidth=2, edgecolor=colour, facecolor="none"
        ))
        w, h = size
        lbl = f"#{i+1}  {w:.0f}x{h:.0f}  {angle:.1f}°  ({area:,} px)"
        if is_outlier:
            lbl += "  ⚠ outlier"
        bbox_ax.text(
            center[0], center[1] - 20, lbl,
            color=colour, fontsize=7, fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.45, pad=1, edgecolor="none"),
        )

    # ── Panel 4: watershed on outliers (or same as panel 3) ──────────
    ws_ax = axs.flat[4]
    has_outliers = bool(outlier_analysis and outlier_analysis['outliers'])
    if has_outliers and watershed_img is not None:
        ws_ax.imshow(watershed_img)
        ws_ax.set_title("Watershed on outliers")
    else:
        ws_ax.imshow(overlay)
        ws_ax.set_title(f"No outliers — edges + bboxes ({len(bboxes)} instruments)")
        for i, bbox in enumerate(bboxes):
            center, size, angle, area = bbox
            is_outlier = id(bbox) in outlier_bboxes
            colour = "red" if is_outlier else "lime"
            box_points = np.int32(cv.boxPoints(((center[0], center[1]), size, angle)))
            ws_ax.add_patch(mpatches.Polygon(
                box_points, closed=True,
                linewidth=2, edgecolor=colour, facecolor="none"
            ))
    ws_ax.axis("off")

    # ── Panel 5: per-bbox stats table ────────────────────────────────
    text_ax = axs.flat[5]
    text_ax.set_xticks([])
    text_ax.set_yticks([])
    for spine in text_ax.spines.values():
        spine.set_visible(False)
    text_ax.set_facecolor("#f0f0f0")
    text_ax.set_title("BBox stats", fontsize=9)

    # Build id → (entry, label) lookup from outlier_analysis
    bbox_info = {}
    if outlier_analysis:
        for entry in outlier_analysis.get('outliers', []):
            bbox_info[id(entry['bbox'])] = (entry, 'outlier')
        for entry in outlier_analysis.get('normal', []):
            bbox_info[id(entry['bbox'])] = (entry, 'normal')

    nan = float('nan')
    header = f"  {'#':>6}  {'area_sc':>7}  {'edge_sc':>7}  {'fill_sc':>7}  {'grade':>6}  label"
    sep    = "  " + "─" * (len(header) - 2)
    lines  = [header, sep]

    if outlier_analysis:
        med_a = outlier_analysis.get('median_area_score', nan)
        med_e = outlier_analysis.get('median_edge_score', nan)
        med_f = outlier_analysis.get('median_fill_score', nan)
        med_g = outlier_analysis.get('median_grade',      nan)
        thr_g = outlier_analysis.get('grade_cutoff',       nan)
        lines.append(f"  {'median':>6}  {med_a:>7.2f}  {med_e:>7.2f}  {med_f:>7.2f}  {med_g:>6.2f}  —")
        lines.append(f"  {'thresh':>6}  {'':>7}  {'':>7}  {'':>7}  {'>' + f'{thr_g:.2f}':>6}  —")
        lines.append(sep)

    for i, bbox in enumerate(bboxes):
        bid = id(bbox)
        if bid in bbox_info:
            entry, label = bbox_info[bid]
            a_sc  = entry.get('area_score', nan)
            e_sc  = entry.get('edge_score', nan)
            f_sc  = entry.get('fill_score', nan)
            grade = entry.get('grade',      nan)
        else:
            a_sc = e_sc = f_sc = grade = nan
            label = '?'
        lines.append(f"  #{i + 1:<5}  {a_sc:>7.2f}  {e_sc:>7.2f}  {f_sc:>7.2f}  {grade:>6.2f}  {label}")

    # --- Secondary filter table ---
    lines.append("")
    if outlier_analysis:
        prim      = outlier_analysis.get('primary_candidates', [])
        sec_cut   = outlier_analysis.get('secondary_cutoff',    0.0)
        max_cg    = outlier_analysis.get('max_candidate_grade', 0.0)
        sec_ratio = outlier_analysis.get('secondary_ratio',     nan)
        outlier_ids = {id(e['bbox']) for e in outlier_analysis.get('outliers', [])}
        bbox_id_to_idx = {id(b): i for i, b in enumerate(bboxes)}

        triggered = len(prim) >= 2
        trigger_lbl = "TRIGGERED" if triggered else f"not triggered ({len(prim)} primary candidate(s))"
        lines.append(f"  Secondary filter — {trigger_lbl}")

        if triggered:
            lines.append(f"  max_grade: {max_cg:.2f}   cutoff: {sec_cut:.2f}   ratio: {sec_ratio:.1f}")
            sec_header = f"  {'#':>6}  {'grade':>6}  status"
            sec_sep    = "  " + "─" * (len(sec_header) - 2)
            lines += [sec_header, sec_sep]
            for cand in sorted(prim, key=lambda e: e['grade'], reverse=True):
                bid   = id(cand['bbox'])
                idx   = bbox_id_to_idx.get(bid, -1)
                lbl   = "OUTLIER" if bid in outlier_ids else "demoted"
                lines.append(f"  #{idx + 1:<5}  {cand['grade']:>6.2f}  {lbl}")

    text_ax.text(
        0.05, 0.95, "\n".join(lines),
        transform=text_ax.transAxes,
        ha='left', va='top',
        fontsize=8, family='monospace',
        color='black',
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=96, bbox_inches='tight')
    buf.seek(0)
    img = cv.imdecode(np.frombuffer(buf.getvalue(), dtype=np.uint8), cv.IMREAD_COLOR)
    plt.close()

    try:
        import subprocess, re
        xrandr_out = subprocess.run(['xrandr', '--current'], capture_output=True, text=True).stdout
        m = re.search(r'(\d+)x(\d+)\+0\+0', xrandr_out)
        sw, sh = (int(m.group(1)), int(m.group(2))) if m else (1920, 1080)
    except Exception:
        sw, sh = 1920, 1080

    h, w = img.shape[:2]
    scale = min((sw - 80) / w, (sh - 80) / h, 1.0)
    if scale < 1.0:
        img = cv.resize(img, (int(w * scale), int(h * scale)), interpolation=cv.INTER_AREA)

    cv.namedWindow("Debug", cv.WINDOW_NORMAL)
    cv.imshow("Debug", img)
    cv.resizeWindow("Debug", img.shape[1], img.shape[0])
    cv.waitKey(0)
    cv.destroyAllWindows()


# ------------------------------------------------------------------ #
#  Entry point                                                        #
# ------------------------------------------------------------------ #

def _process_image(image_path, cfg, debugging=False):
    """Run the full pipeline on a single image. Returns the standard 5-tuple."""
    img_bgr = cv.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    img_rgb = cv.resize(img_rgb, cfg.image_dims, interpolation=cv.INTER_AREA)
    image_label = os.path.basename(image_path)
    print(f"\n[debug] {image_path}")

    roi_crop, roi_mask, roi_bbox         = get_ROI_from_color(img_rgb, cfg)
    binary_mask                          = binarize_image(roi_crop, cfg)
    tray_masked, tray_mask, _            = get_tray_crop(roi_crop, binary_mask, cfg)
    reflection_mask                      = detect_specular_reflections(tray_masked, cfg)
    tray_no_bg                           = remove_blue_background(tray_masked, cfg)

    seg_binary, bboxes, outlier_analysis, filter_stats = segment_instruments(tray_no_bg, cfg)

    print(f"  {len(bboxes)} instrument(s)  [scenario: {outlier_analysis['scenario']}]")
    outlier_bboxes = {id(e['bbox']) for e in outlier_analysis['outliers']}
    for i, bbox in enumerate(bboxes):
        center, size, angle, area = bbox
        cx, cy = center
        w, h = size
        flag = "  ⚠ OUTLIER" if id(bbox) in outlier_bboxes else ""
        print(f"    #{i+1}  center=({cx:.1f},{cy:.1f})  size={w:.1f}x{h:.1f}  angle={angle:.1f}°  area={area:,} px{flag}")

    if debugging:
        watershed_img = apply_watershed_to_outliers(seg_binary, outlier_analysis)
        visualise_results(tray_masked, tray_no_bg, seg_binary, bboxes,
                          reflection_mask, image_label=image_label,
                          outlier_analysis=outlier_analysis,
                          filter_stats=filter_stats,
                          img_rgb=img_rgb, roi_bbox=roi_bbox,
                          watershed_img=watershed_img)

    return tray_no_bg, seg_binary, bboxes, outlier_analysis, filter_stats


def main(debugging=False, image_path=None):
    """Run segmentation on one image (random if image_path is None) or all images
    in ./Trays when image_path is the string 'all'."""
    cfg = PreprocessConfig()

    if image_path == 'all':
        all_paths = sorted(
            os.path.join(wd, f)
            for wd, _, files in os.walk("./Trays")
            for f in files
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
        )
        if not all_paths:
            raise FileNotFoundError("No images found in ./Trays")
        print(f"Processing {len(all_paths)} image(s)…")
        results = []
        for path in all_paths:
            try:
                results.append(_process_image(path, cfg, debugging=debugging))
            except Exception as e:
                print(f"  [error] {path}: {e}")
        return results

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

    return _process_image(image_path, cfg, debugging=debugging)


if __name__ == "__main__":
    IMAGE_PATH ='all' #"./Trays/IMG_3347.jpg"#'all'  # None = random, 'all' = every image in ./Trays, or a specific path
    main(debugging=True, image_path=IMAGE_PATH)
