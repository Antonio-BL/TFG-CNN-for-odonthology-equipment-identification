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
from preprocess import get_ROI_from_color, binarize_image, get_tray_crop, remove_blue_background


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
#  Step 2 — Convex hull approximation                                #
# ------------------------------------------------------------------ #

def _apply_convex_hulls(binary):
    """Draw the convex hull outline of each contour onto the existing mask.

    Preserves the original binary pixels and adds the hull edges on top,
    connecting broken edge points without filling the interior."""
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    hull_mask = binary.copy()
    for cnt in contours:
        hull = cv.convexHull(cnt)
        cv.drawContours(hull_mask, [hull], -1, 255, thickness=2)
    return hull_mask


# ------------------------------------------------------------------ #
#  Step 3 — Contour detection and bounding boxes                     #
# ------------------------------------------------------------------ #

def _find_bboxes(binary, min_area):
    """Return a list of (x, y, w, h, area) for each external contour
    whose area exceeds min_area."""
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv.boundingRect(cnt)
        bboxes.append((x, y, w, h, int(area)))
    # Largest first
    bboxes.sort(key=lambda b: b[4], reverse=True)
    return bboxes


# ------------------------------------------------------------------ #
#  Step 4 — Median-area outlier filter                               #
# ------------------------------------------------------------------ #

def _filter_by_median_area(bboxes, threshold):
    """Discard bounding boxes whose area is below `threshold` * median area.

    Args:
        bboxes:    list of (x, y, w, h, area).
        threshold: fraction of the median area used as the lower bound.

    Returns:
        Filtered list, same format, same ordering.
    """
    if not bboxes:
        return bboxes
    median_area = np.median([b[4] for b in bboxes])
    min_area    = threshold * median_area
    return [b for b in bboxes if b[4] >= min_area]


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
    seg_binary  = _apply_convex_hulls(seg_binary)
    bboxes      = _find_bboxes(seg_binary, cfg.seg_min_contour_area)
    bboxes      = _filter_by_median_area(bboxes, cfg.seg_median_area_threshold)
    return seg_binary, bboxes


# ------------------------------------------------------------------ #
#  Visualisation                                                      #
# ------------------------------------------------------------------ #

def visualise_results(tray_no_bg, seg_binary, bboxes):
    """Plot: background-removed image | Otsu binary | bounding boxes with sizes."""
    fig, axs = plt.subplots(1, 3, figsize=(21, 7))

    axs[0].imshow(tray_no_bg)
    axs[0].set_title("Background removed (H only)")
    axs[0].axis("off")

    axs[1].imshow(seg_binary, cmap="gray")
    axs[1].set_title("Otsu binary mask")
    axs[1].axis("off")

    axs[2].imshow(tray_no_bg)
    axs[2].set_title(f"Bounding boxes ({len(bboxes)} instruments)")
    axs[2].axis("off")

    for i, (x, y, w, h, area) in enumerate(bboxes):
        rect = mpatches.Rectangle(
            (x, y), w, h,
            linewidth=2, edgecolor="lime", facecolor="none"
        )
        axs[2].add_patch(rect)
        axs[2].text(
            x, y - 6,
            f"#{i+1}  {w}x{h}  ({area:,} px)",
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
    tray_no_bg                           = remove_blue_background(tray_masked, cfg)

    # Segmentation
    seg_binary, bboxes = segment_instruments(tray_no_bg, cfg)

    print(f"Found {len(bboxes)} instrument(s):")
    for i, (x, y, w, h, area) in enumerate(bboxes):
        print(f"  #{i+1}  bbox=({x}, {y}, {w}, {h})  area={area:,} px")

    if debugging:
        visualise_results(tray_no_bg, seg_binary, bboxes)

    return tray_no_bg, seg_binary, bboxes


if __name__ == "__main__":
    main(debugging=True)
