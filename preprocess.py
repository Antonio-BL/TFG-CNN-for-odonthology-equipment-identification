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
@dataclass
class PreprocessConfig:
    # General
    image_dims: tuple[int, int] = (4284, 5712)                              # Image dimensions during preprocessing (resolution)
    open_kernel_dims: tuple[int, int]= (3, 3)                               # Dimensions of the kernel for morphological open                          
    close_kernel_dims: tuple[int, int]= (20, 20)                            # Dimensions of the kernel for morphological open
    
    # filtering parameters for ALL color filtering actions
    color_filter_method: str = "rgb"                                        # Color system used in the filtering: rgb | hsv
    color_filter_tolerance_rgb: float = 0.5                                 # Tolerance for the filtering using rgb colors
    color_filter_tolerance_hsv: float = 0.15                                # Tolerance for the filtering using hsv colors
  
    # Patch dimensions
    patch_center: int = None                                                # Center around which to get a patch of the image
    patch_size: int = 10                                                    # Length of the sides of the patch

    # ROI detection parameters
    ROI_background_color: tuple[int, int, int] = (30, 90, 170)              # Color of the background of the area of interest in RGB (usually blue). 
    roi_padding: int = 30                                                   # Padding around clusters
    roi_min_area_ratio: float = 0.03                                        # Minimum valid cluster size as a percentage of the image area (default 3% of the total image area)           
    roi_open_kernel_dims: tuple[int, int] = (7, 7)                          # Dimensions of the kernel for morphological open after ROI                         
    roi_close_kernel_dims: tuple[int, int] = (21, 21)                       # Dimensions of the kernel for morphological close after ROI

# ================================================================== #
# Helper functions : used in preprocessing and in segmentation       #
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
# ================================================================== #
# Image information functions                                        #
# ================================================================== #

def get_ROI_from_color(image: np.ndarray, cfg: PreprocessConfig): 
    """
    Detect the ROI corresponding to the paper/tray background using the reference
    color stored in cfg.ROI_background_color.
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
    bg_rgb = np.asarray(cfg.ROI_background_color, dtype=np.uint8).reshape(1, 1, 3)

    # HSV method
    if cfg.color_filter_method.lower().strip() == "hsv":
        image_hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV)
        bg_hsv = cv.cvtColor(bg_rgb, cv.COLOR_RGB2HSV).reshape(3,)

        H0, S0, V0 = bg_hsv.astype(np.int32)
        tol = float(cfg.color_filter_tolerance)

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
        tol = float(cfg.color_filter_tolerance)
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
        patch_dict = get_centered_patch(image=image, cfg=cfg)
        patch = patch_dict["patch"]
        filter_array = get_avg_color(patch, cfg=cfg)  # typically returns RGB (3,)

        # If HSV filtering is selected, convert the reference from RGB -> HSV
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

# %% Main ----------------------------------------------------------
def main():

    # Load default config
    cfg = PreprocessConfig()

    # Load images of the trays
    trays_directory = (r"C:\Users\Antonio\Documents\GITI\TFG\Trays")
    tray_image = load_images(trays_directory, cfg)
    print(f"{len(tray_image)} images loaded.\n")

    # Select random image from folder to debug code
    debug_img = tray_image[np.random.randint(low=0, high=len(tray_image))]

    # NOTE: debug_img is RGB (because load_images converts BGR -> RGB)
    img_rgb = debug_img

    return

# -------------------------------------------------------------------------
if __name__ == "__main__":
    DEBUGGING = True
    main()
    