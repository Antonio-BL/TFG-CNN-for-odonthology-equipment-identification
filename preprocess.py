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
                    get_multi_patches, get_avg_color, binarize_image)

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
    tol = (float(cfg.color_filter_tolerance_hsv)
           if cfg.color_filter_method.lower().strip() == "hsv"
           else float(cfg.color_filter_tolerance_rgb))

    if cfg.color_filter_method.lower().strip() == "hsv":
        image_hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV)
        bg_hsv    = cv.cvtColor(bg_rgb, cv.COLOR_RGB2HSV).reshape(3,)
        H0, S0, V0 = bg_hsv.astype(np.int32)

        dH, dS, dV = int(179*tol), int(255*tol), int(255*tol)
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
        delta = np.array([255.0, 255.0, 255.0]) * tol
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


# ================================================================== #
# Main                                                               #
# ================================================================== #
def main(debugging: bool = False):
    cfg = PreprocessConfig()

    # -- Load images --
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