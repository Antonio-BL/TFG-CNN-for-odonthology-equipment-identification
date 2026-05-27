# segmentation.py
# Core instrument segmentation pipeline for the TFG dental-instrument project.
#
# Pipeline:
#   0. _connect_edges         — morphological close to bridge small edge gaps
#   1. _binarize_tray         — Sauvola local threshold → binary mask
#   2. _find_bboxes           — oriented bboxes from external contours
#   3. _filter_by_median_area — drop implausibly small blobs
#   4. _analyze_bbox_outliers — classify bboxes as normal / fused-instrument outlier
#
# This module is algorithm-only: it does NOT import matplotlib and contains
# no visualisation code.  All plotting lives in visualize.py.
# Watershed splitting of fused instruments lives in watershed.py.

import numpy as np
import cv2 as cv
from skimage.filters import threshold_sauvola

from config import PreprocessConfig


# ------------------------------------------------------------------ #
#  Step 0 — Bridge small gaps between instrument edges               #
# ------------------------------------------------------------------ #

def _connect_edges(tray_no_bg: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Apply a morphological CLOSE on each RGB channel to bridge small edge gaps.

    WHY: Thin instruments, scratched surfaces, and specular-reflection inpainting
    can leave hairline gaps in the colour gradient between an instrument and the
    background.  A CLOSE (dilate → erode) bridges gaps up to half the kernel
    radius without displacing larger structures, making the subsequent Sauvola
    binarization more reliable.

    Args:
        tray_no_bg: RGB image with the blue tray background zeroed out (H, W, 3).
        cfg:        Pipeline configuration; reads cfg.seg_close_kernel_dims.

    Returns:
        RGB image of the same size with small edge gaps bridged.
    """
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, cfg.seg_close_kernel_dims)
    return cv.morphologyEx(tray_no_bg, cv.MORPH_CLOSE, kernel)


# ------------------------------------------------------------------ #
#  Step 1 — Sauvola local binarization                               #
# ------------------------------------------------------------------ #

def _binarize_tray(tray_no_bg: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Convert the background-removed image to a binary foreground mask.

    WHY: A global threshold (e.g. Otsu) fails on images with uneven illumination
    — bright reflections near dark instrument handles, or instrument tips near a
    brightly lit tray border.  Sauvola adapts the threshold per-pixel from the
    local mean and standard deviation, producing stable foreground masks across
    the full tray.  A post-binarization dilate + open cleans residual salt-noise
    while preserving the binary shape of each instrument.

    Args:
        tray_no_bg: RGB image with background removed (H, W, 3).
        cfg:        Pipeline configuration; reads seg_sauvola_window_size,
                    seg_sauvola_k, seg_close_kernel_dims, bin_open_kernel_dims.

    Returns:
        uint8 binary mask (H, W); instruments = 255, background = 0.
    """
    gray = cv.cvtColor(tray_no_bg, cv.COLOR_RGB2GRAY)

    # Sauvola: threshold(x, y) = mean(x,y) + k × std(x,y), adapts per window
    thresh = threshold_sauvola(
        gray,
        window_size=cfg.seg_sauvola_window_size,
        k=cfg.seg_sauvola_k,
    )
    binary = (gray > thresh).astype(np.uint8) * 255

    # Dilate to recover instrument edges that barely fall below the threshold
    dilate_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, cfg.seg_close_kernel_dims)
    binary = cv.morphologyEx(binary, cv.MORPH_DILATE, dilate_kernel)

    # Open to remove isolated salt-noise specks that survived dilation
    open_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, cfg.bin_open_kernel_dims)
    binary = cv.morphologyEx(binary, cv.MORPH_OPEN, open_kernel)

    return binary


# ------------------------------------------------------------------ #
#  Step 2 — Contour detection and oriented bounding boxes            #
# ------------------------------------------------------------------ #

def _find_bboxes(binary: np.ndarray, min_area: int) -> list[tuple]:
    """Fit a minimum-area oriented bounding box to each foreground contour.

    WHY: Instruments are long, thin, and may be rotated at arbitrary angles.
    An axis-aligned rectangle would include too much background and make size
    comparisons unreliable.  cv.minAreaRect returns the tightest oriented box,
    which is invariant to rotation and gives a clean width × height footprint
    regardless of instrument orientation.

    Args:
        binary:   uint8 binary mask (H, W); instruments = 255.
        min_area: Minimum contour area in pixels; smaller blobs are ignored
                  to remove dust specks and inpainting artefacts.

    Returns:
        List of (center, size, angle, contour_area) where:
          center        — (cx, cy) float tuple, bbox centre in image pixels.
          size          — (w, h)  float tuple, oriented-box side lengths (px).
          angle         — rotation in degrees (OpenCV convention).
          contour_area  — raw polygon area (px²) from cv.contourArea.
        Sorted by descending contour_area (largest instrument first).
    """
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < min_area:
            continue
        center, size, angle = cv.minAreaRect(cnt)
        bboxes.append((center, size, angle, int(area)))

    bboxes.sort(key=lambda b: b[3], reverse=True)
    return bboxes


# ------------------------------------------------------------------ #
#  Step 3 — Per-bbox statistics                                      #
# ------------------------------------------------------------------ #

def _compute_bbox_stats(binary: np.ndarray, bbox: tuple) -> dict:
    """Compute geometric and pixel statistics for one oriented bounding box.

    WHY: These metrics characterise each candidate instrument blob and are used
    by _analyze_bbox_outliers to distinguish normal instruments from fused
    (touching) pairs.  Returning a dict keeps the API extensible — future SVM
    features can add entries here without changing any caller signature.

    Args:
        binary: uint8 binary mask (H, W); instruments = 255.
        bbox:   (center, size, angle, contour_area) 4-tuple from _find_bboxes.

    Returns:
        Dict with keys:
          contour_area    — area of the contour polygon (px²).
          bbox_area       — oriented-box area  w × h  (px²).
          perimeter       — oriented-box perimeter  2*(w+h)  (px).
          width, height   — sides of the oriented box (px).
          fill_ratio      — contour_area / bbox_area in [0, 1].
                            Low fill → the blob is irregular or has interior holes.
          positive_pixels — white pixels inside the oriented bbox (from binary).
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


# ------------------------------------------------------------------ #
#  Step 4 — Median-area filter                                       #
# ------------------------------------------------------------------ #

def _filter_by_median_area(
    bboxes: list[tuple],
    binary: np.ndarray,
    threshold: float,
) -> list[tuple]:
    """Discard bboxes whose effective area is below threshold × median area.

    WHY: Tiny reflection artefacts or seams between tray compartments
    occasionally survive binarization and produce spurious small blobs.
    Rather than a fixed pixel-count cutoff (which would be image-size
    dependent), we use a threshold relative to the median area of all detected
    blobs.  Instruments on the same tray are roughly similar in size, so a
    blob at < 20 % of the median is almost certainly noise.

    Metric: positive_pixels / fill_ratio, which equals the oriented-bbox area
    (w × h).  A thin diagonal instrument has a small contour polygon area but
    a large bbox footprint — the metric keeps it while a tiny speck is dropped.

    Args:
        bboxes:    List of (center, size, angle, contour_area) from _find_bboxes.
        binary:    uint8 binary mask used to count positive pixels per bbox.
        threshold: Lower bound as a fraction of the median (e.g. 0.2 = 20 %).

    Returns:
        Filtered list in the same format and ordering as the input.
    """
    if not bboxes:
        return bboxes

    def _effective_area(bbox: tuple) -> float:
        stats = _compute_bbox_stats(binary, bbox)
        fill = stats['fill_ratio']
        return stats['positive_pixels'] / fill if fill > 0 else 0.0

    metrics = [_effective_area(b) for b in bboxes]
    min_val = threshold * float(np.median(metrics))
    return [b for b, m in zip(bboxes, metrics) if m >= min_val]


# ------------------------------------------------------------------ #
#  Step 5 — Outlier (fused-instrument) detection                     #
# ------------------------------------------------------------------ #

def _analyze_bbox_outliers(
    binary: np.ndarray,
    bboxes: list[tuple],
    component_threshold: float = 1.5,
    secondary_ratio: float = 1.5,
    weight_area: float = 0.6,
    weight_fill: float = 0.4,
) -> dict:
    """Classify bounding boxes as normal instruments or fused-instrument outliers.

    WHY: When two instruments touch, their binary blobs merge into one large,
    irregular blob whose bounding box is bigger and less compact than the boxes
    of individual instruments.  We exploit this with a grade that combines the
    relative area (how much bigger is this bbox?) and inverse fill ratio (how
    irregular is it?).  A bbox whose grade is well above the median is almost
    certainly a fused pair.

    Grade formula (scores normalised to [1, ∞) relative to the minimum across
    all bboxes so they are dimensionless and comparable):

        area_score = bbox_area   / min_bbox_area   (larger  → fused)
        fill_score = inv_fill    / min_inv_fill     (sparser → irregular)
        grade      = weight_area × area_score + weight_fill × fill_score

    Primary pass:  flag any bbox with  grade > component_threshold × median_grade.

    Secondary pass (only when ≥ 2 primary candidates):
    Drop candidates with  grade < max_candidate_grade / secondary_ratio.
    This prevents a marginal candidate from triggering the 'multiple' scenario
    when there is really only one true fused blob.

    Scenarios:
      'none'     — no bbox exceeds the grade cutoff.
      'single'   — exactly one bbox survived both passes; one fused pair.
      'multiple' — two or more survived; multiple fused pairs (rare).

    Args:
        binary:              uint8 binary mask (H, W); instruments = 255.
        bboxes:              List of (center, size, angle, contour_area).
        component_threshold: grade > this × median_grade → primary candidate.
        secondary_ratio:     candidate must have grade ≥ max_grade / this.
        weight_area:         Weight for area_score (default 0.6).
        weight_fill:         Weight for fill_score (default 0.4).
                             weight_area + weight_fill should equal 1.0.

    Returns:
        Dict with keys:
          'scenario'            — 'none' | 'single' | 'multiple'.
          'outliers'            — list of entry dicts for outlier bboxes.
          'normal'              — list of entry dicts for non-outlier bboxes.
          'median_area_score'   — float.
          'median_fill_score'   — float.
          'median_grade'        — float.
          'grade_cutoff'        — float (component_threshold × median_grade).
          'primary_candidates'  — list of entries before the secondary pass.
          'max_candidate_grade' — float, peak grade among primary candidates.
          'secondary_cutoff'    — float (max_grade / secondary_ratio, or 0.0).
          'secondary_ratio'     — float (echoed from the parameter).

        Each entry dict contains:
          'bbox'        — original (center, size, angle, contour_area) tuple.
          'area_score'  — float.
          'fill_score'  — float.
          'grade'       — float.
          + all keys from _compute_bbox_stats.
    """
    if not bboxes:
        return {
            'scenario':            'none',
            'outliers':            [],
            'normal':              [],
            'median_area_score':   0.0,
            'median_fill_score':   0.0,
            'median_grade':        0.0,
            'grade_cutoff':        0.0,
            'primary_candidates':  [],
            'max_candidate_grade': 0.0,
            'secondary_cutoff':    0.0,
            'secondary_ratio':     secondary_ratio,
        }

    # Raw per-bbox metrics
    bbox_areas = [b[1][0] * b[1][1] for b in bboxes]
    inv_fills  = [
        (b[1][0] * b[1][1]) / b[3] if b[3] > 0 else 0.0
        for b in bboxes
    ]

    # Normalise to [1, ∞) so scores are dimensionless
    min_bbox_area = max(float(np.min(bbox_areas)), 1.0)
    min_inv_fill  = max(float(np.min(inv_fills)),  1e-6)

    area_scores = [ba  / min_bbox_area for ba  in bbox_areas]
    fill_scores = [ivf / min_inv_fill  for ivf in inv_fills]
    grades      = [
        weight_area * a + weight_fill * f
        for a, f in zip(area_scores, fill_scores)
    ]

    median_area_score = float(np.median(area_scores))
    median_fill_score = float(np.median(fill_scores))
    median_grade      = float(np.median(grades))
    grade_cutoff      = component_threshold * median_grade

    # Primary pass: flag bboxes above the grade cutoff
    outliers, normal = [], []
    for bbox, a_sc, f_sc, grade in zip(bboxes, area_scores, fill_scores, grades):
        stats = _compute_bbox_stats(binary, bbox)
        entry = {'bbox': bbox, 'area_score': a_sc, 'fill_score': f_sc,
                 'grade': grade, **stats}
        if grade > grade_cutoff:
            outliers.append(entry)
        else:
            normal.append(entry)

    # Secondary pass: when ≥ 2 primary candidates, demote those far below the peak
    primary_candidates   = list(outliers)
    max_candidate_grade  = max((e['grade'] for e in outliers), default=0.0)
    secondary_cutoff_val = 0.0

    if len(outliers) >= 2:
        secondary_cutoff_val = max_candidate_grade / secondary_ratio
        kept    = [e for e in outliers if e['grade'] >= secondary_cutoff_val]
        demoted = [e for e in outliers if e['grade'] <  secondary_cutoff_val]
        normal  += demoted
        outliers = kept

    if   len(outliers) == 0: scenario = 'none'
    elif len(outliers) == 1: scenario = 'single'
    else:                    scenario = 'multiple'

    return {
        'scenario':            scenario,
        'outliers':            outliers,
        'normal':              normal,
        'median_area_score':   median_area_score,
        'median_fill_score':   median_fill_score,
        'median_grade':        median_grade,
        'grade_cutoff':        grade_cutoff,
        'primary_candidates':  primary_candidates,
        'max_candidate_grade': max_candidate_grade,
        'secondary_cutoff':    secondary_cutoff_val,
        'secondary_ratio':     secondary_ratio,
    }


# ------------------------------------------------------------------ #
#  Public API                                                         #
# ------------------------------------------------------------------ #

def segment_instruments(
    tray_no_bg: np.ndarray,
    cfg: PreprocessConfig,
) -> tuple[np.ndarray, list[tuple], dict]:
    """Detect instrument bounding boxes from the background-removed tray image.

    This is the main public entry point for the segmentation module.  It runs
    the core pipeline and returns ALL detected bboxes together with the outlier
    classification.  Watershed splitting of fused instruments is an optional
    separate step — call watershed.watershed_split() on the outlier bboxes if
    you need to break them apart.

    Pipeline:
      0. _connect_edges         — bridge small edge gaps (morphological close).
      1. _binarize_tray         — Sauvola local threshold → binary mask.
      2. _find_bboxes           — oriented bboxes for every contour above min area.
      3. _filter_by_median_area — drop implausibly small blobs.
      4. _analyze_bbox_outliers — classify as normal / fused-instrument outlier.

    Args:
        tray_no_bg: RGB image from remove_blue_background (H, W, 3); the blue
                    background and outer border are zeroed out.
        cfg:        PreprocessConfig.

    Returns:
        seg_binary:       uint8 binary mask (H, W); instruments = 255.
        bboxes:           Full list of (center, size, angle, area) 4-tuples —
                          normal + outlier bboxes, NOT yet split by watershed.
        outlier_analysis: Dict from _analyze_bbox_outliers; contains 'outliers'
                          and 'normal' entry lists for downstream use.
    """
    tray_closed      = _connect_edges(tray_no_bg, cfg)
    seg_binary       = _binarize_tray(tray_closed, cfg)
    bboxes           = _find_bboxes(seg_binary, cfg.seg_min_contour_area)
    bboxes           = _filter_by_median_area(
        bboxes, seg_binary, cfg.seg_median_area_threshold
    )
    outlier_analysis = _analyze_bbox_outliers(
        seg_binary,
        bboxes,
        component_threshold=cfg.seg_outlier_grade_threshold,
        secondary_ratio=cfg.seg_outlier_secondary_ratio,
        weight_area=cfg.seg_outlier_weight_area,
        weight_fill=cfg.seg_outlier_weight_fill,
    )
    return seg_binary, bboxes, outlier_analysis
