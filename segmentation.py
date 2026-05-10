# segmentation.py
# Segments surgical instruments from the background-removed tray image.
# Pipeline: binarize (Otsu) -> contour detection -> bounding boxes -> visualise

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

from config     import PreprocessConfig
from preprocess import (get_ROI_from_color, binarize_image, get_tray_crop,
                        remove_blue_background, detect_specular_reflections)


# ------------------------------------------------------------------ #
#  Step 0 — Morphological close to connect nearby edges              #
# ------------------------------------------------------------------ #

def _connect_edges(tray_no_bg, cfg):
    """Apply a morphological close on each RGB channel to bridge small
    gaps between instrument edges before binarization."""
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, cfg.seg_close_kernel_dims)
    return cv.morphologyEx(tray_no_bg, cv.MORPH_CLOSE, kernel)


# ------------------------------------------------------------------ #
#  Step 1 — Otsu binarization                                        #
# ------------------------------------------------------------------ #

def _binarize_tray(tray_no_bg):
    """Grayscale + Otsu threshold on the background-removed image.
    Both the outer tray border and the blue background are already zeroed,
    so Otsu separates instrument pixels from the black canvas cleanly."""
    gray = cv.cvtColor(tray_no_bg, cv.COLOR_RGB2GRAY)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return binary


# ------------------------------------------------------------------ #
#  Step 2 — KNN clustering of contours                               #
# ------------------------------------------------------------------ #

def _apply_knn_clustering(binary, distance_threshold=50, size_ratio_threshold=0.5):
    """Gently merge only nearby small contour fragments.

    This approach only merges small contours (below median area) that are
    very close to each other, avoiding aggressive over-merging. Uses distance-based
    clustering rather than k-nearest neighbors for finer control.

    Args:
        binary: Binary mask (H, W) with instrument pixels = 255.
        distance_threshold: Maximum distance between centroids to consider merging (pixels).
        size_ratio_threshold: Only merge contours below this ratio of median area.

    Returns:
        Merged binary mask with gently merged contours.
    """
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if len(contours) <= 1:
        result = binary.copy()
        if contours:
            cv.drawContours(result, contours, -1, 255, thickness=cv.FILLED)
        return result

    # Compute centroids and areas of each contour
    centroids = []
    areas = []
    valid_indices = []

    for idx, cnt in enumerate(contours):
        M = cv.moments(cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            area = cv.contourArea(cnt)
            centroids.append([cx, cy])
            areas.append(area)
            valid_indices.append(idx)

    if not centroids:
        return binary

    centroids = np.array(centroids, dtype=np.float32)
    areas = np.array(areas)
    median_area = np.median(areas)

    # Identify small contours (fragments) to consider for merging
    small_mask = areas < (size_ratio_threshold * median_area)

    # Build distance-based clusters for small contours only
    clusters = {}
    visited = set()
    cluster_id = 0

    for i in range(len(centroids)):
        if i in visited or not small_mask[i]:
            # Large contours are never merged, each forms its own cluster
            clusters[cluster_id] = {i}
            visited.add(i)
            cluster_id += 1
            continue

        cluster = set()
        queue = [i]

        while queue:
            curr = queue.pop(0)
            if curr in visited:
                continue
            visited.add(curr)
            cluster.add(curr)

            # Find nearby contours within distance threshold
            curr_centroid = centroids[curr]
            for j in range(len(centroids)):
                if j in visited or not small_mask[j]:
                    continue

                # Calculate distance to this contour
                dist = np.linalg.norm(centroids[j] - curr_centroid)

                if dist <= distance_threshold:
                    queue.append(j)

        clusters[cluster_id] = cluster
        cluster_id += 1

    # Draw merged contours from clusters
    result = np.zeros_like(binary)
    for cluster_indices in clusters.values():
        # Merge contours in the cluster
        merged_contour_points = []
        for idx in cluster_indices:
            original_idx = valid_indices[idx]
            merged_contour_points.extend(contours[original_idx].reshape(-1, 2))

        if merged_contour_points:
            merged_contour_points = np.array(merged_contour_points, dtype=np.int32)
            # Draw convex hull of merged points
            hull = cv.convexHull(merged_contour_points)
            cv.drawContours(result, [hull], -1, 255, thickness=cv.FILLED)

    return result


# ------------------------------------------------------------------ #
#  Step 3 — Contour detection and bounding boxes                     #
# ------------------------------------------------------------------ #

def _find_bboxes(binary, min_area):
    """Return a list of oriented bounding boxes for each external contour
    whose area exceeds min_area.

    Returns:
        list of (center, size, angle, area) where:
        - center: (x, y) tuple
        - size: (width, height) tuple
        - angle: rotation angle in degrees
        - area: contour area in pixels
    """
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < min_area:
            continue
        # Use minAreaRect for oriented bounding box
        rect = cv.minAreaRect(cnt)
        center, size, angle = rect
        bboxes.append((center, size, angle, int(area)))
    # Largest first
    bboxes.sort(key=lambda b: b[3], reverse=True)
    return bboxes


# ------------------------------------------------------------------ #
#  Step 4 — Median-area outlier filter                               #
# ------------------------------------------------------------------ #

def _filter_by_median_area(bboxes, threshold):
    """Discard bounding boxes whose area is below `threshold` * median area.

    Args:
        bboxes:    list of (center, size, angle, area).
        threshold: fraction of the median area used as the lower bound.

    Returns:
        Filtered list, same format, same ordering.
    """
    if not bboxes:
        return bboxes
    median_area = np.median([b[3] for b in bboxes])
    min_area    = threshold * median_area
    return [b for b in bboxes if b[3] >= min_area]


# ------------------------------------------------------------------ #
#  Step 5 — Outlier detection (median-area filter)                   #
# ------------------------------------------------------------------ #

def _compute_bbox_stats(binary, bbox):
    """Compute per-bbox statistics used for outlier detection and future SVM features.

    Returns a dict with:
      contour_area    – area of the contour polygon (px²)
      bbox_area       – oriented bounding-box area  w × h  (px²)
      width, height   – sides of the oriented bbox  (px)
      fill_ratio      – contour_area / bbox_area  (0–1)
      positive_pixels – white pixels inside the oriented bbox from binary mask
    """
    center, size, angle, contour_area = bbox
    w, h = size
    bbox_area = float(w * h)

    box_pts = cv.boxPoints(((center[0], center[1]), size, angle))
    box_pts = np.int32(box_pts)
    roi_mask = np.zeros(binary.shape[:2], dtype=np.uint8)
    cv.fillPoly(roi_mask, [box_pts], 255)
    positive_pixels = int(np.count_nonzero(cv.bitwise_and(binary, roi_mask)))

    return {
        'contour_area':    contour_area,
        'bbox_area':       bbox_area,
        'width':           float(w),
        'height':          float(h),
        'fill_ratio':      contour_area / bbox_area if bbox_area > 0 else 0.0,
        'positive_pixels': positive_pixels,
    }


def _analyze_bbox_outliers(binary, bboxes, area_ratio_threshold=2.0):
    """Classify bboxes into normal and outlier groups based on their area
    relative to the median.

    A bbox is an outlier when  contour_area > area_ratio_threshold × median_area.

    Scenarios
    ---------
    'none'     – no bbox exceeds the threshold; instruments are uniformly sized.
    'single'   – exactly one outlier; likely two touching instruments fused.
    'multiple' – two or more outliers; multiple fused or unusually large regions.

    Returns
    -------
    dict with keys:
      'scenario'    : 'none' | 'single' | 'multiple'
      'outliers'    : list of {'bbox': ..., **stats}  – outlier entries
      'normal'      : list of {'bbox': ..., **stats}  – non-outlier entries
      'median_area' : float
    """
    if not bboxes:
        return {'scenario': 'none', 'outliers': [], 'normal': [], 'median_area': 0.0}

    median_area = float(np.median([b[3] for b in bboxes]))
    cutoff      = area_ratio_threshold * median_area

    outliers, normal = [], []
    for bbox in bboxes:
        stats = _compute_bbox_stats(binary, bbox)
        entry = {'bbox': bbox, **stats}
        (outliers if bbox[3] > cutoff else normal).append(entry)

    if   len(outliers) == 0: scenario = 'none'
    elif len(outliers) == 1: scenario = 'single'
    else:                    scenario = 'multiple'

    return {
        'scenario':    scenario,
        'outliers':    outliers,
        'normal':      normal,
        'median_area': median_area,
    }


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
    seg_binary       = _binarize_tray(tray_closed)
    seg_binary       = _apply_knn_clustering(seg_binary, distance_threshold=50, size_ratio_threshold=0.5)
    bboxes           = _find_bboxes(seg_binary, cfg.seg_min_contour_area)
    bboxes           = _filter_by_median_area(bboxes, cfg.seg_median_area_threshold)
    outlier_analysis = _analyze_bbox_outliers(seg_binary, bboxes, cfg.seg_outlier_area_ratio)
    return seg_binary, bboxes, outlier_analysis


# ------------------------------------------------------------------ #
#  Visualisation                                                      #
# ------------------------------------------------------------------ #

def visualise_results(tray_masked, tray_no_bg, seg_binary, bboxes,
                      reflection_mask=None, image_label=None, outlier_analysis=None):
    """Plot: original tray | reflections | background-removed | Otsu binary | bounding boxes."""
    n_plots = 5 if reflection_mask is not None else 4
    ncols = 3
    nrows = -(-n_plots // ncols)  # ceiling division
    fig, axs = plt.subplots(nrows, ncols, figsize=(18, 6 * nrows))
    if image_label:
        fig.suptitle(image_label, fontsize=13, fontweight="bold")
    axs.flat[0].imshow(tray_masked)
    axs.flat[0].set_title("Original tray masked")
    axs.flat[0].axis("off")

    idx = 1
    if reflection_mask is not None:
        axs.flat[idx].imshow(reflection_mask, cmap="gray")
        axs.flat[idx].set_title("Specular reflections")
        axs.flat[idx].axis("off")
        idx += 1

    axs.flat[idx].imshow(tray_no_bg)
    axs.flat[idx].set_title("Background removed (H only)")
    axs.flat[idx].axis("off")
    idx += 1

    axs.flat[idx].imshow(seg_binary, cmap="gray")
    axs.flat[idx].set_title("Otsu binary mask")
    axs.flat[idx].axis("off")
    idx += 1

    bbox_ax = axs.flat[idx]
    bbox_ax.imshow(tray_no_bg)
    bbox_ax.set_title(f"Bounding boxes ({len(bboxes)} instruments)")
    bbox_ax.axis("off")

    outlier_bboxes = set()
    if outlier_analysis and outlier_analysis['outliers']:
        outlier_bboxes = {id(e['bbox']) for e in outlier_analysis['outliers']}

    for i, bbox in enumerate(bboxes):
        center, size, angle, area = bbox
        is_outlier = id(bbox) in outlier_bboxes
        colour = "red" if is_outlier else "lime"
        box_points = cv.boxPoints(((center[0], center[1]), size, angle))
        box_points = np.int32(box_points)
        polygon = mpatches.Polygon(
            box_points, closed=True,
            linewidth=2, edgecolor=colour, facecolor="none"
        )
        bbox_ax.add_patch(polygon)
        w, h = size
        label = f"#{i+1}  {w:.0f}x{h:.0f}  {angle:.1f}°  ({area:,} px)"
        if is_outlier:
            label += "  ⚠ outlier"
        bbox_ax.text(
            center[0], center[1] - 20, label,
            color=colour, fontsize=7, fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.45, pad=1, edgecolor="none"),
        )

    # Hide any unused cells
    for ax in list(axs.flat)[n_plots:]:
        ax.axis("off")

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

def main(debugging=False, image_path=None):
    cfg = PreprocessConfig()

    if image_path is None:
        all_paths = [
            os.path.join(wd, f)
            for wd, _, files in os.walk("./Trays")
            for f in files
        ]
        if not all_paths:
            raise FileNotFoundError("No images found in ./Trays")
        image_path = all_paths[np.random.randint(0, len(all_paths))]

    img_bgr = cv.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    img_rgb = cv.resize(img_rgb, cfg.image_dims, interpolation=cv.INTER_AREA)
    image_label = os.path.basename(image_path)
    print(f"[debug] Loaded image: {image_path}")

    # Preprocessing pipeline
    roi_crop, roi_mask, roi_bbox         = get_ROI_from_color(img_rgb, cfg)
    binary_mask                          = binarize_image(roi_crop, cfg)
    tray_masked, tray_mask, _            = get_tray_crop(roi_crop, binary_mask, cfg)
    reflection_mask                      = detect_specular_reflections(tray_masked, cfg)
    tray_no_bg                           = remove_blue_background(tray_masked, cfg)

    # Segmentation
    seg_binary, bboxes, outlier_analysis = segment_instruments(tray_no_bg, cfg)

    print(f"Found {len(bboxes)} instrument(s)  [outlier scenario: {outlier_analysis['scenario']}]")
    outlier_bboxes = {id(e['bbox']) for e in outlier_analysis['outliers']}
    for i, bbox in enumerate(bboxes):
        center, size, angle, area = bbox
        cx, cy = center
        w, h = size
        flag = "  ⚠ OUTLIER" if id(bbox) in outlier_bboxes else ""
        print(f"  #{i+1}  center=({cx:.1f}, {cy:.1f})  size={w:.1f}x{h:.1f}  angle={angle:.1f}°  area={area:,} px{flag}")

    if debugging:
        visualise_results(tray_masked, tray_no_bg, seg_binary, bboxes,
                          reflection_mask, image_label=image_label,
                          outlier_analysis=outlier_analysis)

    return tray_no_bg, seg_binary, bboxes, outlier_analysis


if __name__ == "__main__":
    IMAGE_PATH = None # "./Trays"   # set to a path string to load a specific image, e.g. 
    main(debugging=True, image_path=IMAGE_PATH)
