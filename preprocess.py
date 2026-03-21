# Loads and preprocess images before segmentation.
#   1. Load Images in RGB format:                           load_images
#   2. Get the working area (ROI) based on background       get_ROI_from_color
#   3. Get the edges of the image                           edge_Laplace
#   4. Binarizes the image                                  binarize_image

# ================================================================== #
# Basic dependencies                                                 #
# ================================================================== #
import os
import platform 
from typing import Optional
import numpy as np 
import pandas as pd 

import seaborn as sns
import matplotlib.pyplot as plt

from dataclasses import dataclass
# ================================================================== #
# Image processing dependencies                                      #
# ================================================================== #
import cv2 as cv
from scipy.signal import convolve2d

# ================================================================== #
# Linux support                                                      #
# ================================================================== #
# force x11
if platform.system() == "Linux":
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

# ================================================================== #
#  Preprocess Configuration class                                    # 
# ================================================================== #
from config import PreprocessConfig

# ================================================================== #
#  Helper Functions and Utilities                                    # 
# ================================================================== #
from utils import load_images
from utils import edge_Laplace
from utils import open_close_cleanup
from utils import get_multi_patches
from utils import get_avg_color
from utils import binarize_image

# ================================================================== #
#  Local Functions                                                   # 
# ================================================================== #

def get_ROI_from_color(image: np.ndarray, cfg: PreprocessConfig): 
    """
    Detect the ROI corresponding to the paper/tray background using the reference
    color given by get_avg_color
    Uses HSV

    Args:
        image: RGB image, shape (H, W, 3)
        cfg: PreprocessConfig

    Returns:
        roi_crop: cropped RGB image containing the ROI
        roi_mask: binary mask uint8(H, W), ROI region = 255
        roi_bbox: tuple (x, y, w, h)
    """
    assert image is not None, "image is None"
    assert image.ndim == 3 and image.shape[2] == 3, (
        f"Expected RGB image (H,W,3), got shape {image.shape}"
    )

    H_img, W_img = image.shape[:2]

    # 1) Build background mask from cfg.ROI_background_color
    # ---------------------------------------------------------
    bg_rgb = np.asarray(get_avg_color(get_multi_patches(image, cfg), cfg), dtype=np.uint8).reshape(1, 1, 3)

    # Tolerance
    if cfg.color_filter_method.lower().strip() == "hsv":
        tol = float(cfg.color_filter_tolerance_hsv)
    else:
        tol = float(cfg.color_filter_tolerance_rgb)

    # HSV method
    if cfg.color_filter_method.lower().strip() == "hsv":
        image_hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV)
        bg_hsv = cv.cvtColor(bg_rgb, cv.COLOR_RGB2HSV).reshape(3,)

        H0, S0, V0 = bg_hsv.astype(np.int32)

        dH = int(179 * tol)
        dS = int(255 * tol)
        dV = int(255 * tol)

        s_low = max(S0 - dS, 0)
        s_up  = min(S0 + dS, 255)
        v_low = max(V0 - dV, 0)
        v_up  = min(V0 + dV, 255)

        h_low = H0 - dH
        h_up  = H0 + dH

        # Hue wrap-around
        if h_low < 0 or h_up > 179:
            lowerA = np.array([0, s_low, v_low], dtype=np.uint8)
            upperA = np.array([min(h_up, 179), s_up, v_up], dtype=np.uint8)
            maskA = cv.inRange(image_hsv, lowerA, upperA)

            lowerB = np.array([max(h_low + 180, 0), s_low, v_low], dtype=np.uint8)
            upperB = np.array([179, s_up, v_up], dtype=np.uint8)
            maskB = cv.inRange(image_hsv, lowerB, upperB)

            bg_mask = cv.bitwise_or(maskA, maskB)
        else:
            lower = np.array([h_low, s_low, v_low], dtype=np.uint8)
            upper = np.array([h_up, s_up, v_up], dtype=np.uint8)
            bg_mask = cv.inRange(image_hsv, lower, upper)

    # RGB method
    else:
        ref = bg_rgb.reshape(3,).astype(np.float32)
        delta = np.array([255.0, 255.0, 255.0], dtype=np.float32) * tol

        lower = np.clip(ref - delta, 0, 255).astype(np.uint8)
        upper = np.clip(ref + delta, 0, 255).astype(np.uint8)

        bg_mask = cv.inRange(image, lower, upper)


    # 2) Morphological cleanup on background mask
    # ---------------------------------------------------------
    bg_mask = open_close_cleanup(bg_mask, cfg)

    # 3) Keep only the largest connected component: Clustering
    # ---------------------------------------------------------
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(bg_mask, connectivity=8)

    if num_labels <= 1:
        raise ValueError("No ROI background region detected.")

    min_area = int(cfg.roi_min_area_ratio * H_img * W_img)

    best_label = None
    best_area = -1

    for lbl in range(1, num_labels):
        area = stats[lbl, cv.CC_STAT_AREA]
        if area >= min_area and area > best_area:
            best_area = area
            best_label = lbl

    if best_label is None:
        raise ValueError(
            f"No ROI component large enough. Try reducing roi_min_area_ratio "
            f"or increasing color_filter_tolerance."
        )

    roi_mask = np.zeros_like(bg_mask)
    roi_mask[labels == best_label] = 255

    # Optional extra closing after selecting the component
    close_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, cfg.roi_close_kernel_dims)
    roi_mask = cv.morphologyEx(roi_mask, cv.MORPH_CLOSE, close_kernel)

    # 4) Bounding box + padding
    # ---------------------------------------------------------
    x, y, w, h = cv.boundingRect(cv.findNonZero(roi_mask))

    pad = int(cfg.roi_padding)
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(W_img, x + w + pad)
    y1 = min(H_img, y + h + pad)

    roi_crop = image[y0:y1, x0:x1]
    roi_bbox = (x0, y0, x1 - x0, y1 - y0)

    return roi_crop, roi_mask, roi_bbox

# ================================================================== #
#  MAIN                                                              # 
# ================================================================== #

# %% Main ----------------------------------------------------------
def main():
    cfg = PreprocessConfig()

    tray_images = load_images("./Trays", cfg)
    if not tray_images:
        raise FileNotFoundError("No images found in ./Trays")

    img_rgb = tray_images[np.random.randint(0, len(tray_images))]

    roi_crop, roi_mask, roi_bbox = get_ROI_from_color(img_rgb, cfg)

    # -- visualize ROI --
    x0, y0, w, h = roi_bbox
    viz = img_rgb.copy()
    cv.rectangle(viz, (x0, y0), (x0+w, y0+h), (0,255,0), thickness=8)

    fig, axs = plt.subplots(1, 2, figsize=(14,6))
    axs[0].imshow(viz);      axs[0].set_title("ROI bounding box"); axs[0].axis("off")
    axs[1].imshow(roi_crop); axs[1].set_title("ROI crop");         axs[1].axis("off")
    plt.tight_layout(); plt.show()

    # -- binarization --
    patches = get_multi_patches(roi_crop, cfg)
    ref_rgb = get_avg_color(patches, cfg)
    cfg.color_filter_method = "hsv"
    mask_hsv = binarize_image(roi_crop, cfg)

    plt.figure(figsize=(6,6))
    plt.imshow(mask_hsv, cmap="gray")
    plt.title("Binarized (HSV)"); plt.axis("off"); plt.show()

    return

# -------------------------------------------------------------------------
if __name__ == "__main__":
    DEBUGGING = True
    main()
    