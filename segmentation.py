# segmentation.py
# Segments surgical instruments from the background-removed tray image.
# Pipeline: binarize (Otsu) -> contour detection -> bounding boxes -> visualise

import os
import platform
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

if platform.system() == "Linux":
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

from config     import PreprocessConfig
from utils      import load_images
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
    tray_closed = _connect_edges(tray_no_bg, cfg)
    seg_binary  = _binarize_tray(tray_closed)
    seg_binary  = _apply_knn_clustering(seg_binary, distance_threshold=50, size_ratio_threshold=0.5)
    bboxes      = _find_bboxes(seg_binary, cfg.seg_min_contour_area)
    bboxes      = _filter_by_median_area(bboxes, cfg.seg_median_area_threshold)
    return seg_binary, bboxes


# ------------------------------------------------------------------ #
#  Visualisation                                                      #
# ------------------------------------------------------------------ #

def visualise_results(tray_masked, tray_no_bg, seg_binary, bboxes, reflection_mask=None):
    """Plot: original tray | reflections | background-removed | Otsu binary | bounding boxes."""
    n_plots = 5 if reflection_mask is not None else 4
    figsize = (35, 7) if reflection_mask is not None else (28, 7)
    fig, axs = plt.subplots(1, n_plots, figsize=figsize)

    axs[0].imshow(tray_masked)
    axs[0].set_title("Original tray masked")
    axs[0].axis("off")

    if reflection_mask is not None:
        axs[1].imshow(reflection_mask, cmap="gray")
        axs[1].set_title("Specular reflections")
        axs[1].axis("off")
        offset = 1
    else:
        offset = 0

    axs[1 + offset].imshow(tray_no_bg)
    axs[1 + offset].set_title("Background removed (H only)")
    axs[1 + offset].axis("off")

    axs[2 + offset].imshow(seg_binary, cmap="gray")
    axs[2 + offset].set_title("Otsu binary mask")
    axs[2 + offset].axis("off")

    axs[3 + offset].imshow(tray_no_bg)
    axs[3 + offset].set_title(f"Bounding boxes ({len(bboxes)} instruments)")
    axs[3 + offset].axis("off")

    for i, (center, size, angle, area) in enumerate(bboxes):
        # Convert minAreaRect to corner points for drawing
        box_points = cv.boxPoints(((center[0], center[1]), size, angle))
        box_points = np.int32(box_points)

        # Draw rotated rectangle
        polygon = mpatches.Polygon(
            box_points, closed=True,
            linewidth=2, edgecolor="lime", facecolor="none"
        )
        axs[3 + offset].add_patch(polygon)

        # Label at top-left corner
        w, h = size
        axs[3 + offset].text(
            center[0], center[1] - 20,
            f"#{i+1}  {w:.0f}x{h:.0f}  {angle:.1f}°  ({area:,} px)",
            color="lime", fontsize=7, fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.45, pad=1, edgecolor="none"),
        )

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------ #
#  Entry point                                                        #
# ------------------------------------------------------------------ #

def main(debugging=False):
    cfg = PreprocessConfig()

    tray_images = load_images("./Trays", cfg)
    if not tray_images:
        raise FileNotFoundError("No images found in ./Trays")

    img_rgb = tray_images[np.random.randint(0, len(tray_images))]

    # Preprocessing pipeline
    roi_crop, roi_mask, roi_bbox         = get_ROI_from_color(img_rgb, cfg)
    binary_mask                          = binarize_image(roi_crop, cfg)
    tray_masked, tray_mask, _            = get_tray_crop(roi_crop, binary_mask, cfg)
    reflection_mask                      = detect_specular_reflections(tray_masked, cfg)
    tray_no_bg                           = remove_blue_background(tray_masked, cfg)

    # Segmentation
    seg_binary, bboxes = segment_instruments(tray_no_bg, cfg)

    print(f"Found {len(bboxes)} instrument(s):")
    for i, (center, size, angle, area) in enumerate(bboxes):
        cx, cy = center
        w, h = size
        print(f"  #{i+1}  center=({cx:.1f}, {cy:.1f})  size={w:.1f}x{h:.1f}  angle={angle:.1f}°  area={area:,} px")

    if debugging:
        visualise_results(tray_masked, tray_no_bg, seg_binary, bboxes, reflection_mask)

    return tray_no_bg, seg_binary, bboxes


if __name__ == "__main__":
    main(debugging=True)
