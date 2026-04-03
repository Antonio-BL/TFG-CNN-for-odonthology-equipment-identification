# preprocessing.py
# Loads and preprocesses images before segmentation.
#   1. Load Images in RGB format:                           load_images
#   2. Get the working area (ROI) based on background:      get_ROI_from_color
#   3. Binarizes the image:                                 binarize_image

# ================================================================== #
# Basic dependencies                                                 #
# ================================================================== #
import os
import platform
import numpy as np
import matplotlib.pyplot as plt

# ================================================================== #
# Image processing dependencies                                      #
# ================================================================== #
import cv2 as cv

# ================================================================== #
# Linux support                                                      #
# ================================================================== #
if platform.system() == "Linux":
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

# ================================================================== #
# Project imports                                                    #
# ================================================================== #
from config import PreprocessConfig
from utils  import (load_images, open_close_cleanup,
                    get_multi_patches, get_avg_color)

# ================================================================== #
# Local functions                                                    #
# ================================================================== #
def get_ROI_from_color(image: np.ndarray, cfg: PreprocessConfig):
    """
    Detects the ROI corresponding to the tray background using
    an adaptive color reference from get_avg_color.
    Uses HSV color space.

    Args:
        image: RGB image, shape (H, W, 3)
        cfg:   PreprocessConfig

    Returns:
        roi_crop: cropped RGB image containing the ROI
        roi_mask: binary mask uint8 (H, W), ROI region = 255
        roi_bbox: tuple (x0, y0, w, h)
    """
    assert image is not None, "image is None"
    assert image.ndim == 3 and image.shape[2] == 3, \
        f"Expected RGB image (H,W,3), got shape {image.shape}"

    H_img, W_img = image.shape[:2]

    # 1) Estimate background color adaptively
    # ---------------------------------------------------------
    bg_rgb = np.asarray(
        get_avg_color(get_multi_patches(image, cfg), cfg),
        dtype=np.uint8
    ).reshape(1, 1, 3)

    # 2) Build background mask
    # ---------------------------------------------------------
    if cfg.color_filter_method.lower().strip() == "hsv":
        image_hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV)
        bg_hsv    = cv.cvtColor(bg_rgb, cv.COLOR_RGB2HSV).reshape(3,)
        H0, S0, V0 = bg_hsv.astype(np.int32)

        dH = int(179 * cfg.color_filter_tolerance_h)
        dS = int(255 * cfg.color_filter_tolerance_s)
        dV = int(255 * cfg.color_filter_tolerance_v)
        s_low, s_up = max(S0-dS, 0),   min(S0+dS, 255)
        v_low, v_up = max(V0-dV, 0),   min(V0+dV, 255)
        h_low, h_up = H0-dH, H0+dH

        if h_low < 0 or h_up > 179:
            maskA = cv.inRange(image_hsv,
                               np.array([0,             s_low, v_low], dtype=np.uint8),
                               np.array([min(h_up, 179),s_up,  v_up],  dtype=np.uint8))
            maskB = cv.inRange(image_hsv,
                               np.array([max(h_low+180, 0), s_low, v_low], dtype=np.uint8),
                               np.array([179,                s_up,  v_up],  dtype=np.uint8))
            bg_mask = cv.bitwise_or(maskA, maskB)
        else:
            bg_mask = cv.inRange(image_hsv,
                                 np.array([h_low, s_low, v_low], dtype=np.uint8),
                                 np.array([h_up,  s_up,  v_up],  dtype=np.uint8))
    else:
        ref   = bg_rgb.reshape(3,).astype(np.float32)
        delta = np.array([255.0, 255.0, 255.0]) * float(cfg.color_filter_tolerance_rgb)
        bg_mask = cv.inRange(image,
                             np.clip(ref-delta, 0, 255).astype(np.uint8),
                             np.clip(ref+delta, 0, 255).astype(np.uint8))

    # 3) Morphological cleanup
    # ---------------------------------------------------------
    bg_mask = open_close_cleanup(bg_mask, cfg)

    # 4) Keep only the largest connected component
    # ---------------------------------------------------------
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(
        bg_mask, connectivity=8
    )

    if num_labels <= 1:
        raise ValueError("No ROI background region detected.")

    min_area   = int(cfg.roi_min_area_ratio * H_img * W_img)
    best_label, best_area = None, -1

    for lbl in range(1, num_labels):
        area = stats[lbl, cv.CC_STAT_AREA]
        if area >= min_area and area > best_area:
            best_area, best_label = area, lbl

    if best_label is None:
        raise ValueError(
            "No ROI component large enough. "
            "Try reducing roi_min_area_ratio or increasing color_filter_tolerance."
        )

    roi_mask = np.zeros_like(bg_mask)
    roi_mask[labels == best_label] = 255

    close_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, cfg.roi_close_kernel_dims)
    roi_mask = cv.morphologyEx(roi_mask, cv.MORPH_CLOSE, close_kernel)

    # 5) Bounding box + padding
    # ---------------------------------------------------------
    x, y, w, h = cv.boundingRect(cv.findNonZero(roi_mask))
    pad = int(cfg.roi_padding)
    x0  = max(0,      x - pad)
    y0  = max(0,      y - pad)
    x1  = min(W_img,  x + w + pad)
    y1  = min(H_img,  y + h + pad)

    roi_crop = image[y0:y1, x0:x1]
    roi_bbox = (x0, y0, x1-x0, y1-y0)

    return roi_crop, roi_mask, roi_bbox


def normalize_illumination_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Normalises uneven illumination in an RGB image using CLAHE applied to the
    luminance channel of the YCrCb colour space.

    CLAHE (Contrast Limited Adaptive Histogram Equalisation) redistributes pixel
    intensities locally within a grid of tiles, boosting contrast in dark regions
    (such as cast shadows) without amplifying noise — unlike global histogram
    equalisation. The clip_limit parameter caps the contrast amplification per tile
    to prevent over-enhancement.

    YCrCb separates luminance (Y) from chroma (Cr, Cb), so applying CLAHE only to
    channel 0 (Y) normalises brightness across the image while leaving the colour
    information in Cr and Cb untouched. This is critical here because the HSV
    filtering that follows relies on accurate hue and saturation values: if CLAHE
    were applied to all channels it would distort those values and undermine the
    colour-based background detection.

    By evening out the V (brightness) component before the HSV filter sees the
    image, shadows on the tray background — which lower V without changing H — are
    brought back into the brightness range of well-lit background pixels. This
    prevents shadowed background regions from being misclassified as tools.

    Args:
        image:      RGB image, shape (H, W, 3), dtype uint8.
        clip_limit: Contrast amplification limit per CLAHE tile. Higher values
                    produce stronger equalisation but may introduce artefacts.
        tile_grid:  (rows, cols) number of tiles CLAHE divides the image into.
                    Smaller grids → more global; larger grids → more local.

    Returns:
        Illumination-normalised RGB image, shape (H, W, 3), dtype uint8.
    """
    image_ycrcb = cv.cvtColor(image, cv.COLOR_RGB2YCrCb)
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    image_ycrcb[:, :, 0] = clahe.apply(image_ycrcb[:, :, 0])
    return cv.cvtColor(image_ycrcb, cv.COLOR_YCrCb2RGB).astype(np.uint8)


def binarize_image(image: np.ndarray, cfg: PreprocessConfig, filter_array=None) -> np.ndarray:
    """
    Produces a binary mask that isolates surgical instruments from the tray background.

    Processing steps:
      1. Illumination normalisation via CLAHE (see normalize_illumination_clahe) so
         that shadowed background pixels are not misclassified as tools.
      2. Background colour estimation from multi-patch sampling (when filter_array
         is not provided).
      3. Background masking using per-channel HSV tolerances (H, S, V each have an
         independent tolerance configured in PreprocessConfig), or a uniform RGB
         tolerance when cfg.color_filter_method == "rgb".
      4. Morphological open + close to remove small noise blobs.

    Args:
        image:        RGB image to binarize, shape (H, W, 3), dtype uint8.
        cfg:          Configuration parameters (filter method, tolerances, kernels,
                      CLAHE settings).
        filter_array: Reference colour for background detection. When None it is
                      computed automatically from the average colour of peripheral
                      patches. For HSV mode this must be an HSV triplet; for RGB
                      mode an RGB triplet.

    Returns:
        filtered_image: uint8 binary mask, shape (H, W). Tools = 255, background = 0.
    """
    # 1) Normalise illumination so shadows do not fool the colour filter
    image_normalised = normalize_illumination_clahe(
        image,
        clip_limit=cfg.clahe_clip_limit,
        tile_grid=cfg.clahe_tile_grid,
    )

    valid_methods = ["hsv", "rgb"]
    filter_method = cfg.color_filter_method.lower().strip()

    if filter_array is None:
        patches = get_multi_patches(image=image_normalised, cfg=cfg)
        filter_array = get_avg_color(patches, cfg=cfg)
        if filter_method == "hsv":
            ref_rgb = np.asarray(filter_array, dtype=np.uint8).reshape(1, 1, 3)
            filter_array = cv.cvtColor(ref_rgb, cv.COLOR_RGB2HSV).reshape(3,)

    assert filter_method in valid_methods, (
        f"Method {cfg.color_filter_method} not recognized: choose between HSV or RGB \n"
    )

    filter_array = np.asarray(filter_array).reshape(-1)
    assert filter_array.size == 3, f"filter_array must contain 3 values, got shape {np.asarray(filter_array).shape}"

    if filter_method == "rgb":
        tol = float(cfg.color_filter_tolerance_rgb)
        ref = filter_array.astype(np.float32)
        delta = np.array([255.0, 255.0, 255.0], dtype=np.float32) * tol
        lower = np.clip(ref - delta, 0, 255).astype(np.uint8)
        upper = np.clip(ref + delta, 0, 255).astype(np.uint8)
        mask_bg = cv.inRange(image_normalised, lower, upper)
    else:
        image_hsv = cv.cvtColor(image_normalised, cv.COLOR_RGB2HSV)
        H, S, V = filter_array.astype(np.int32)
        dH = int(179 * cfg.color_filter_tolerance_h)
        dS = int(255 * cfg.color_filter_tolerance_s)
        dV = int(255 * cfg.color_filter_tolerance_v)
        s_low, s_up = max(S - dS, 0), min(S + dS, 255)
        v_low, v_up = max(V - dV, 0), min(V + dV, 255)
        h_low, h_up = H - dH, H + dH

        if h_low < 0 or h_up > 179:
            maskA = cv.inRange(image_hsv,
                               np.array([0,             s_low, v_low], dtype=np.uint8),
                               np.array([min(h_up, 179), s_up,  v_up], dtype=np.uint8))
            maskB = cv.inRange(image_hsv,
                               np.array([max(h_low + 180, 0), s_low, v_low], dtype=np.uint8),
                               np.array([179,                  s_up,  v_up], dtype=np.uint8))
            mask_bg = cv.bitwise_or(maskA, maskB)
        else:
            mask_bg = cv.inRange(image_hsv,
                                 np.array([h_low, s_low, v_low], dtype=np.uint8),
                                 np.array([h_up,  s_up,  v_up],  dtype=np.uint8))

    filtered_image = cv.bitwise_not(mask_bg)

    open_kernel  = cv.getStructuringElement(cv.MORPH_ELLIPSE, cfg.open_kernel_dims)
    close_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, cfg.close_kernel_dims)
    filtered_image = cv.morphologyEx(filtered_image, cv.MORPH_OPEN,  open_kernel)
    filtered_image = cv.morphologyEx(filtered_image, cv.MORPH_CLOSE, close_kernel)

    return filtered_image


# ================================================================== #
# Main                                                               #
# ================================================================== #
def main(debugging: bool = False):
    cfg = PreprocessConfig()

    # -- Load images --
    # debug=True (set in PreprocessConfig): loads one random image.
    # debug=False: loads all images, using a disk cache at ./cache/images.pkl.
    tray_images = load_images("./Trays", cfg)
    if not tray_images:
        raise FileNotFoundError("No images found in ./Trays")

    img_rgb = tray_images[np.random.randint(0, len(tray_images))]

    # -- ROI detection --
    roi_crop, roi_mask, roi_bbox = get_ROI_from_color(img_rgb, cfg)    
    # -- Binarization --
    binary_mask = binarize_image(roi_crop, cfg)

    # -- Debug visualizations --
    if debugging:
        # ROI on original image
        x0, y0, w, h = roi_bbox
        viz = img_rgb.copy()
        cv.rectangle(viz, (x0, y0), (x0+w, y0+h), (0, 255, 0), thickness=8)

        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        axs[0].imshow(viz)
        axs[0].set_title("ROI bounding box")
        axs[0].axis("off")

        axs[1].imshow(roi_crop)
        axs[1].set_title("ROI crop")
        axs[1].axis("off")

        axs[2].imshow(binary_mask, cmap="gray")
        axs[2].set_title("Binary mask")
        axs[2].axis("off")

        plt.tight_layout()
        plt.show()

    return roi_crop, binary_mask, roi_bbox

if __name__ == "__main__":
    main(debugging=True)