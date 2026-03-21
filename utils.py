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
#  Global Functions: Helper Functions                                # 
# ================================================================== #
def load_images(path: str, cfg: PreprocessConfig) -> list[np.ndarray]:
    '''
    Reads images in specified directories and loads them in an array.
    Args: 
        path: Path to scan for images.
        cfg: Configuration parameters.
    Returns: 
        Array containing scanned images.
    Raises: 
        FileNotFoundError: if the directory does not exist
    ''' 
    if not os.path.isdir(path): 
        raise FileNotFoundError(f"{path} is not a directory \n")

    images = []
    for working_directory, directory, file in os.walk(path):
        for f in file:
            img_path = os.path.join(working_directory, f)
            img= cv.imread(img_path)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            if img is None:
                print(f" Warning: Image {img_path} has been skipped: It is empty \n")
                continue
         
            img = cv.resize(img, cfg.image_dims, interpolation=cv.INTER_AREA)
            images.append(img)
               
    return images

def edge_Laplace(image: np.array ) -> np.ndarray:
    '''
    Edge detection applying the Laplace convolution
    Args: 
        image: image to apply Laplace edge detection on.
    Returns: 
        edges_image: image containing the edges.
    '''
    # Laplacian matrix
    L = 0.25 * np.array([[0, -1, 0],[-1, 4, -1], [0, -1, 0]])
    print(f"shape L: {L.shape}")
    print(f"image shape: {image.shape}")
    edges_image =  convolve2d(image, L, mode="same", boundary="symm")

    return edges_image

def open_close_cleanup(image: np.ndarray, cfg: PreprocessConfig) -> np.ndarray: 
    """
    Applies morphological opening and then close to clean up the iamge from possible noise.
    Uses the defined opening and closing kernels.
    Arguments: 
        image: np.ndarray, image to be filtered. RGB format.
        cfg: Configuration parameters.
    Returns:
        cleaned_image: np.ndarray image after closing and opening.
    """
    open_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, cfg.roi_open_kernel_dims)
    close_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, cfg.roi_close_kernel_dims)

    cleaned_image = cv.morphologyEx(image, cv.MORPH_OPEN, open_kernel)
    cleaned_image = cv.morphologyEx(cleaned_image, cv.MORPH_CLOSE, close_kernel)

    return cleaned_image


def get_multi_patches(image: np.ndarray, cfg: PreprocessConfig) -> list[np.ndarray]:
    """
    Extract multiple patches around the image (avoid the center).
    Returns a list of RGB patches.
    Arguments: 
        image: np.ndarray. RGB Image to get patches from.
        cfg: Configuration class.
    Returns: 
        patches: List of patches of the image.
    """
    H, W = image.shape[:2]

    # Patch side length in pixels (cfg.patch_size is % of min dimension)
    side = int(min(H, W) * (cfg.patch_size / 100.0))
    side = max(side, 20)

    # Patch centers: 8 positions around the borders (no center)
    centers = [
        (int(0.15*W), int(0.15*H)), (int(0.50*W), int(0.15*H)), (int(0.85*W), int(0.15*H)),
        (int(0.15*W), int(0.50*H)),                           (int(0.85*W), int(0.50*H)),
        (int(0.15*W), int(0.85*H)), (int(0.50*W), int(0.85*H)), (int(0.85*W), int(0.85*H)),
    ]

    patches = []
    for cx, cy in centers:
        x0 = max(cx - side // 2, 0)
        x1 = min(cx + side // 2, W)
        y0 = max(cy - side // 2, 0)
        y1 = min(cy + side // 2, H)

        patch = image[y0:y1, x0:x1]
        patches.append(patch)

    return patches
# ================================================================== #
#  Global Functions: Getting image characteristics                   # 
# ================================================================== #

def get_avg_color(patches: list[np.ndarray], cfg: PreprocessConfig) -> np.ndarray:
    """
    Robust background color estimation from multiple RGB patches.
    Returns RGB uint8 (3,).
    Gets avg color after ensuring S and V values are within range. 
    Arguments: 
        patches: list of patches of an image to ger an average color of.
        cfg: Configuration class.
    Returns: 
        ref_rgb: Average color in rgb format.
    """
    hsv_samples = []

    H_lim, S_lim, V_lim = cfg.color_filter_hsv_limits

    for patch in patches:
        hsv = cv.cvtColor(patch, cv.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)

        H, S, V = hsv[:, 0], hsv[:, 1], hsv[:, 2]

        keep = np.ones(len(hsv), dtype=bool)

        # Apply  limits
        if H_lim[0] is not None:
            keep &= H >= H_lim[0]       # Lower H Limit
        if H_lim[1] is not None:
            keep &= H <= H_lim[1]       # Upper H Limit

        if S_lim[0] is not None:
            keep &= S >= S_lim[0]       # Lower S Limit
        if S_lim[1] is not None:
            keep &= S <= S_lim[1]       # Upper S Limit

        if V_lim[0] is not None:
            keep &= V >= V_lim[0]       # Lower V Limit
        if V_lim[1] is not None:
            keep &= V <= V_lim[1]       # Upper V Limit

        hsv = hsv[keep]

        if hsv.shape[0] < 0.10 * patch.size / 3:
            continue

        hsv_samples.append(hsv)

    if len(hsv_samples) == 0:
        # Fallback: global median on the full image (still HSV robust)
        # (you can pass the full image instead if you want)
        return np.array(cfg.ROI_background_color, dtype=np.uint8)

    hsv_all = np.vstack(hsv_samples)

    # Median is robust to outliers
    H_med = np.median(hsv_all[:, 0])
    S_med = np.median(hsv_all[:, 1])
    V_med = np.median(hsv_all[:, 2])

    ref_hsv = np.array([H_med, S_med, V_med], dtype=np.uint8).reshape(1, 1, 3)
    ref_rgb = cv.cvtColor(ref_hsv, cv.COLOR_HSV2RGB).reshape(3,)

    return ref_rgb.astype(np.uint8)


def binarize_image(image: np.ndarray, cfg: PreprocessConfig, filter_array: Optional[np.ndarray]=None) -> np.ndarray:
    '''
    Filters the image to obtain a binarized image by using color. Can filter in RGB color system and HSV.

    Args:
        image: np.ndarray, image to be filtered. RGB format.
        cfg: Configuration parameters.
        filter_array: array containing the filter reference values (RGB or HSV depending on method).
                      If None, it is computed from the average color of a centered patch.
    Returns:
        filtered_image: np.ndarray uint8(H, W) containing the binarized mask (tool=255, background=0).
    '''
    # Initialize values and read arguments
    valid_methods = ["hsv", "rgb"]
    filtered_image = image  # will be overwritten with the final binary mask
    filter_method = cfg.color_filter_method.lower().strip()

    # --- Get reference color if not provided ---
    if filter_array is None:
        patches = get_multi_patches(image=image, cfg=cfg)
        filter_array = get_avg_color(patches, cfg=cfg)  # returns RGB
        if filter_method == "hsv":
            ref_rgb = np.asarray(filter_array, dtype=np.uint8).reshape(1, 1, 3)
            filter_array = cv.cvtColor(ref_rgb, cv.COLOR_RGB2HSV).reshape(3,)

    # --- Validate method ---
    assert filter_method in valid_methods, (
        f"Method {cfg.color_filter_method} not recognized: choose between HSV or RGB \n"
    )

    # --- Validate filter_array shape ---
    # Accept (3,), (3,1) or (1,3) and normalize to (3,)
    filter_array = np.asarray(filter_array).reshape(-1)
    assert filter_array.size == 3, f"filter_array must contain 3 values, got shape {np.asarray(filter_array).shape}"

    # --- Build mask of background pixels close to reference color ---
    if filter_method == "rgb":

        tol = float(cfg.color_filter_tolerance_rgb)
        # filter_array is [R,G,B] in [0..255]
        ref = filter_array.astype(np.float32)
        delta = np.array([255.0, 255.0, 255.0], dtype=np.float32) * tol

        lower = np.clip(ref - delta, 0, 255).astype(np.uint8)
        upper = np.clip(ref + delta, 0, 255).astype(np.uint8)

        mask_bg = cv.inRange(image, lower, upper)

    else:  # hsv
        # filter_array is [H,S,V] where H in [0..179], S,V in [0..255]
        image_hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV)
        tol = float(cfg.color_filter_tolerance_hsv)

        H, S, V = filter_array.astype(np.int32)
        dH = int(179 * tol)
        dS = int(255 * tol)
        dV = int(255 * tol)

        s_low = max(S - dS, 0)
        s_up  = min(S + dS, 255)
        v_low = max(V - dV, 0)
        v_up  = min(V + dV, 255)

        h_low = H - dH
        h_up  = H + dH

        # Hue is circular -> handle wrap-around by combining two ranges if needed
        if h_low < 0 or h_up > 179:
            lowerA = np.array([0, s_low, v_low], dtype=np.uint8)
            upperA = np.array([min(h_up, 179), s_up, v_up], dtype=np.uint8)
            maskA = cv.inRange(image_hsv, lowerA, upperA)

            lowerB = np.array([max(h_low + 180, 0), s_low, v_low], dtype=np.uint8)
            upperB = np.array([179, s_up, v_up], dtype=np.uint8)
            maskB = cv.inRange(image_hsv, lowerB, upperB)

            mask_bg = cv.bitwise_or(maskA, maskB)
        else:
            lower = np.array([h_low, s_low, v_low], dtype=np.uint8)
            upper = np.array([h_up,  s_up,  v_up], dtype=np.uint8)
            mask_bg = cv.inRange(image_hsv, lower, upper)

    # --- Invert so that tools/foreground are white (255) ---
    filtered_image = cv.bitwise_not(mask_bg)

    # --- Morphological cleanup ---
    open_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, cfg.open_kernel_dims)
    close_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, cfg.close_kernel_dims)

    filtered_image = cv.morphologyEx(filtered_image, cv.MORPH_OPEN, open_kernel)
    filtered_image = cv.morphologyEx(filtered_image, cv.MORPH_CLOSE, close_kernel)

    return filtered_image

