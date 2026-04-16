# preprocess.py
# Pipeline: load_images -> get_ROI_from_color -> binarize_image -> get_tray_crop -> remove_blue_background

import os
import platform
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

if platform.system() == "Linux":
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

from config import PreprocessConfig
from utils import (open_close_cleanup, get_multi_patches, get_avg_color)


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

def _build_hsv_mask(image_hsv, ref_hsv, cfg, tol_h=None, tol_s=None, tol_v=None):
    """Build an inRange mask in HSV with per-channel tolerances.

    tol_h/s/v override cfg.color_filter_tolerance_h/s/v when provided,
    allowing per-call tuning without mutating the config.
    """
    H0, S0, V0 = ref_hsv.astype(np.int32)
    dH = int(179 * (tol_h if tol_h is not None else cfg.color_filter_tolerance_h))
    dS = int(255 * (tol_s if tol_s is not None else cfg.color_filter_tolerance_s))
    dV = int(255 * (tol_v if tol_v is not None else cfg.color_filter_tolerance_v))

    s_low, s_up = max(S0 - dS, 0), min(S0 + dS, 255)
    v_low, v_up = max(V0 - dV, 0), min(V0 + dV, 255)
    h_low, h_up = H0 - dH, H0 + dH

    if h_low < 0 or h_up > 179:
        mask_a = cv.inRange(
            image_hsv,
            np.array([0, s_low, v_low], dtype=np.uint8),
            np.array([min(h_up, 179), s_up, v_up], dtype=np.uint8),
        )
        mask_b = cv.inRange(
            image_hsv,
            np.array([max(h_low + 180, 0), s_low, v_low], dtype=np.uint8),
            np.array([179, s_up, v_up], dtype=np.uint8),
        )
        return cv.bitwise_or(mask_a, mask_b)

    return cv.inRange(
        image_hsv,
        np.array([h_low, s_low, v_low], dtype=np.uint8),
        np.array([h_up, s_up, v_up], dtype=np.uint8),
    )


def _build_rgb_mask(image, ref_rgb, cfg):
    """Build an inRange mask in RGB with uniform tolerance."""
    ref = ref_rgb.astype(np.float32)
    delta = np.array([255.0, 255.0, 255.0]) * float(cfg.color_filter_tolerance_rgb)
    return cv.inRange(
        image,
        np.clip(ref - delta, 0, 255).astype(np.uint8),
        np.clip(ref + delta, 0, 255).astype(np.uint8),
    )


def _build_color_mask(image, ref_rgb, cfg):
    """Dispatch to HSV or RGB mask builder based on cfg."""
    method = cfg.color_filter_method.lower().strip()
    if method == "hsv":
        image_hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV)
        ref_hsv = cv.cvtColor(
            ref_rgb.reshape(1, 1, 3), cv.COLOR_RGB2HSV
        ).reshape(3)
        return _build_hsv_mask(image_hsv, ref_hsv, cfg)
    if method == "rgb":
        return _build_rgb_mask(image, ref_rgb.reshape(3), cfg)
    raise ValueError(f"Unknown color_filter_method: {cfg.color_filter_method!r}")


# ------------------------------------------------------------------ #
#  Step 1 — ROI detection                                            #
# ------------------------------------------------------------------ #

def get_ROI_from_color(image, cfg):
    """Detect the ROI (blue tray background) and return its crop, mask and bbox.

    Args:
        image: RGB uint8 (H, W, 3).
        cfg:   PreprocessConfig.

    Returns:
        roi_crop: cropped RGB image.
        roi_mask: binary mask (H, W), ROI = 255.
        roi_bbox: (x0, y0, w, h).
    """
    assert image is not None and image.ndim == 3 and image.shape[2] == 3

    H_img, W_img = image.shape[:2]

    # Adaptive background-color estimate
    bg_rgb = np.asarray(
        get_avg_color(get_multi_patches(image, cfg), cfg), dtype=np.uint8
    )

    # Background mask
    bg_mask = _build_color_mask(image, bg_rgb, cfg)
    bg_mask = open_close_cleanup(bg_mask, cfg)

    # Keep only the largest connected component above the area threshold
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(bg_mask, 8)
    if num_labels <= 1:
        raise ValueError("No ROI background region detected.")

    min_area = int(cfg.roi_min_area_ratio * H_img * W_img)
    best_label, best_area = None, -1
    for lbl in range(1, num_labels):
        area = stats[lbl, cv.CC_STAT_AREA]
        if area >= min_area and area > best_area:
            best_area, best_label = area, lbl

    if best_label is None:
        raise ValueError("No ROI component large enough.")

    roi_mask = np.zeros_like(bg_mask)
    roi_mask[labels == best_label] = 255

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, cfg.roi_close_kernel_dims)
    roi_mask = cv.morphologyEx(roi_mask, cv.MORPH_CLOSE, kernel)

    # Bounding box + padding
    x, y, w, h = cv.boundingRect(cv.findNonZero(roi_mask))
    pad = cfg.roi_padding
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(W_img, x + w + pad)
    y1 = min(H_img, y + h + pad)

    return image[y0:y1, x0:x1], roi_mask, (x0, y0, x1 - x0, y1 - y0)


# ------------------------------------------------------------------ #
#  Step 2 — CLAHE illumination normalisation                         #
# ------------------------------------------------------------------ #

def normalize_illumination_clahe(image, clip_limit=2.0, tile_grid=(8, 8)):
    """Apply CLAHE on the luminance channel (YCrCb) to even out shadows.

    Only the Y channel is equalised, so hue and saturation stay intact
    for the downstream HSV colour filter.
    """
    ycrcb = cv.cvtColor(image, cv.COLOR_RGB2YCrCb)
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
    return cv.cvtColor(ycrcb, cv.COLOR_YCrCb2RGB).astype(np.uint8)


# ------------------------------------------------------------------ #
#  Step 2b — Specular reflection detection                           #
# ------------------------------------------------------------------ #

def detect_specular_reflections(image, cfg):
    """Detect specular highlights (bright white reflections on metallic surfaces).

    Uses HSV thresholding:
    - High Value (V > threshold): brightness
    - Low Saturation (S < threshold): white (no color)

    Args:
        image: RGB uint8 (H, W, 3).
        cfg:   PreprocessConfig (uses reflection_v_threshold, reflection_s_threshold).

    Returns:
        reflection_mask: uint8 binary mask (H, W), reflections = 255.
    """
    image_hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV)
    H_ch, S_ch, V_ch = cv.split(image_hsv)

    # High brightness + low saturation = specular reflection
    v_mask = V_ch > cfg.reflection_v_threshold
    s_mask = S_ch < cfg.reflection_s_threshold
    reflection_mask = (v_mask & s_mask).astype(np.uint8) * 255

    return reflection_mask


# ------------------------------------------------------------------ #
#  Step 3 — Binarization                                             #
# ------------------------------------------------------------------ #

def binarize_image(image, cfg, filter_array=None):
    """Produce a binary mask of the blue background (blue = 255, rest = 0).

    Steps:
        1. CLAHE illumination normalisation.
        2. Adaptive background-color estimation (unless filter_array given).
        3. HSV / RGB colour mask.
        4. Morphological open + close cleanup.
    """
    image_norm = normalize_illumination_clahe(
        image,
        clip_limit=cfg.clahe_clip_limit,
        tile_grid=cfg.clahe_tile_grid,
    )

    method = cfg.color_filter_method.lower().strip()

    if filter_array is None:
        patches = get_multi_patches(image_norm, cfg)
        filter_array = get_avg_color(patches, cfg)
        if method == "hsv":
            ref_rgb = np.asarray(filter_array, dtype=np.uint8).reshape(1, 1, 3)
            filter_array = cv.cvtColor(ref_rgb, cv.COLOR_RGB2HSV).reshape(3)

    filter_array = np.asarray(filter_array).reshape(-1)
    assert filter_array.size == 3

    if method == "rgb":
        mask_bg = _build_rgb_mask(image_norm, filter_array, cfg)
    elif method == "hsv":
        image_hsv = cv.cvtColor(image_norm, cv.COLOR_RGB2HSV)
        mask_bg = _build_hsv_mask(
            image_hsv, filter_array, cfg,
            tol_h=cfg.bin_tolerance_h,
            tol_s=cfg.bin_tolerance_s,
            tol_v=cfg.bin_tolerance_v,
        )
    else:
        raise ValueError(f"Unknown color_filter_method: {method!r}")

    open_k = cv.getStructuringElement(cv.MORPH_ELLIPSE, cfg.open_kernel_dims)
    close_k = cv.getStructuringElement(cv.MORPH_ELLIPSE, cfg.close_kernel_dims)
    mask_bg = cv.morphologyEx(mask_bg, cv.MORPH_OPEN, open_k)
    mask_bg = cv.morphologyEx(mask_bg, cv.MORPH_CLOSE, close_k)

    return mask_bg


# ------------------------------------------------------------------ #
#  Step 4 — Tray crop                                                #
# ------------------------------------------------------------------ #

def get_tray_crop(roi_crop, binary_mask, cfg):
    """Isolate the tray from the ROI by contour detection.

    A strong morphological closing is applied to binary_mask before contour
    detection to bridge gaps caused by tools lying on the tray edge: without
    it, a tool that interrupts the blue border creates a notch in the largest
    contour and the tool pixels get masked out.  The closing is done on a
    temporary copy so binary_mask is unchanged for the caller.

    Finds the largest contour in the smoothed mask whose bounding rect is
    smaller than the full ROI (filtering out desk-border artefacts).
    Returns a masked image where only the tray interior is visible.

    Args:
        roi_crop:    RGB image (H, W, 3).
        binary_mask: uint8 mask, blue bg = 255.
        cfg:         PreprocessConfig (uses tray_full_size_tol,
                     tray_edge_close_kernel).

    Returns:
        tray_masked:  RGB image, same shape; non-tray pixels = 0.
        tray_mask:    binary mask (H, W), tray = 255.
        tray_contour: selected contour array (from smoothed mask).
    """
    H, W = roi_crop.shape[:2]
    tol = cfg.tray_full_size_tol

    k = cfg.tray_edge_close_kernel
    close_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k, k))
    smoothed_mask = cv.morphologyEx(binary_mask, cv.MORPH_CLOSE, close_kernel)

    contours, _ = cv.findContours(
        smoothed_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise ValueError("No contours found in binary mask.")

    tray_contour = None
    best_area = -1
    for cnt in contours:
        _, _, cw, ch = cv.boundingRect(cnt)
        if cw >= tol * W and ch >= tol * H:
            continue
        area = cv.contourArea(cnt)
        if area > best_area:
            best_area = area
            tray_contour = cnt

    if tray_contour is None:
        raise ValueError("No valid tray contour found.")

    tray_mask = np.zeros((H, W), dtype=np.uint8)
    cv.drawContours(tray_mask, [tray_contour], -1, 255, thickness=cv.FILLED)

    tray_masked = cv.bitwise_and(roi_crop, roi_crop, mask=tray_mask)
    return tray_masked, tray_mask, tray_contour


# ------------------------------------------------------------------ #
#  Step 5 — Blue background removal (H-only)                        #
# ------------------------------------------------------------------ #

def _repair_specular_reflections(
    image: np.ndarray,
    reflection_mask: np.ndarray,
    inpaint_radius: int = 15,
) -> np.ndarray:
    """Replace specular reflection pixels with values inpainted from their neighbourhood.

    Reflections are bright white (high V, low S in HSV).  Rather than zeroing
    them out, this function estimates the colour each pixel would have without
    the reflection by propagating the surrounding hue and saturation inward
    via the Telea fast-marching inpainting algorithm.  The result blends
    smoothly with the adjacent blue tray and lets the downstream H-only
    background filter classify those pixels correctly.

    Args:
        image:            RGB uint8 (H, W, 3).
        reflection_mask:  Binary mask (H, W), reflections = 255.
        inpaint_radius:   Neighbourhood radius (px) used by cv.inpaint.

    Returns:
        RGB uint8 image with reflection pixels replaced by inpainted values.
    """
    if not np.any(reflection_mask):
        return image.copy()

    bgr          = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    bgr_repaired = cv.inpaint(bgr, reflection_mask, inpaint_radius, cv.INPAINT_TELEA)
    return cv.cvtColor(bgr_repaired, cv.COLOR_BGR2RGB)


def remove_blue_background(tray_masked, cfg, bg_rgb=None):
    """Zero out blue-background pixels using only the H channel.

    Matching on H alone means shadows (which shift S and V but not H)
    are never misclassified as foreground.  If cfg.remove_reflections is
    True, specular highlights (high V, low S) are repaired via inpainting
    before background classification so their corrected hue and saturation
    are used for the decision rather than their washed-out white values.

    Args:
        tray_masked: RGB image, output of get_tray_crop (H, W, 3).
        cfg:         PreprocessConfig.
        bg_rgb:      Background colour as RGB uint8 (3,). When None it is
                     computed from patches of tray_masked.

    Returns:
        RGB image same shape as tray_masked; blue-background pixels = 0.
    """
    image_clahe = normalize_illumination_clahe(
        tray_masked,
        clip_limit=cfg.clahe_clip_limit,
        tile_grid=cfg.clahe_tile_grid,
    )

    if cfg.remove_reflections:
        reflection_mask = detect_specular_reflections(image_clahe, cfg)
        image_clahe     = _repair_specular_reflections(
                              image_clahe, reflection_mask,
                              inpaint_radius=cfg.reflection_inpaint_radius,
                          )

    if bg_rgb is None:
        bg_rgb = get_avg_color(get_multi_patches(image_clahe, cfg), cfg)

    bg_rgb = np.asarray(bg_rgb, dtype=np.uint8).reshape(1, 1, 3)
    H0     = int(cv.cvtColor(bg_rgb, cv.COLOR_RGB2HSV)[0, 0, 0])

    dH    = int(179 * cfg.color_filter_tolerance_h)
    h_low = H0 - dH
    h_up  = H0 + dH

    H_ch = cv.cvtColor(image_clahe, cv.COLOR_RGB2HSV)[:, :, 0]

    if h_low < 0 or h_up > 179:
        is_bg = (H_ch <= min(h_up, 179)) | (H_ch >= max(h_low + 180, 0))
    else:
        is_bg = (H_ch >= h_low) & (H_ch <= h_up)

    result = tray_masked.copy()
    result[is_bg] = 0
    return result


# ------------------------------------------------------------------ #
#  Debug entry point                                                 #
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

    # Pipeline
    roi_crop, roi_mask, roi_bbox          = get_ROI_from_color(img_rgb, cfg)
    binary_mask                           = binarize_image(roi_crop, cfg)
    tray_masked, tray_mask, tray_contour  = get_tray_crop(roi_crop, binary_mask, cfg)
    tray_no_bg                            = remove_blue_background(tray_masked, cfg)

    if debugging:
        x0, y0, w, h = roi_bbox
        viz = img_rgb.copy()
        cv.rectangle(viz, (x0, y0), (x0 + w, y0 + h), (0, 255, 0), thickness=8)

        contours, _ = cv.findContours(
            binary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        contour_viz = cv.cvtColor(roi_crop.copy(), cv.COLOR_RGB2BGR)
        cv.drawContours(contour_viz, contours, -1, (0, 0, 255), thickness=3)
        cv.drawContours(contour_viz, [tray_contour], 0, (0, 255, 0), thickness=5)
        contour_viz = cv.cvtColor(contour_viz, cv.COLOR_BGR2RGB)

        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(image_label, fontsize=13, fontweight="bold")
        titles = [
            "ROI bounding box", "ROI crop",
            "Blue background mask", f"Contours ({len(contours)} total)",
            "Tray masked", "Background removed (H only)",
        ]
        images = [viz, roi_crop, binary_mask, contour_viz, tray_masked, tray_no_bg]
        cmaps  = [None, None, "gray", None, None, None]

        for ax, im, title, cmap in zip(axs.flat, images, titles, cmaps):
            ax.imshow(im, cmap=cmap)
            ax.set_title(title)
            ax.axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    return roi_crop, binary_mask, tray_masked, tray_mask, tray_no_bg, roi_bbox


if __name__ == "__main__":
    IMAGE_PATH = "./Trays\IMG_3354.jpg"   # set to a path string to load a specific image, e.g. "./Trays/IMG_0042.jpg"
    main(debugging=True, image_path=IMAGE_PATH)
