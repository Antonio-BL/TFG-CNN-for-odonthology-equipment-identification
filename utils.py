# ================================================================== #
# Basic dependencies                                                 #
# ================================================================== #
import os
import platform
import numpy as np
import pandas as pd
import pickle

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
_CACHE_FILE = "./cache/images.pkl"

def _collect_image_paths(path: str) -> list[str]:
    paths = []
    for wd, _, files in os.walk(path):
        for f in files:
            paths.append(os.path.join(wd, f))
    return paths

def _load_single(img_path: str, cfg: PreprocessConfig) -> np.ndarray | None:
    img = cv.imread(img_path)
    if img is None:
        return None
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return cv.resize(img, cfg.image_dims, interpolation=cv.INTER_AREA)

def load_images(path: str, cfg: PreprocessConfig) -> list[np.ndarray]:
    '''
    Reads images in specified directories and loads them in an array.

    Behaviour depends on cfg.debug:
      - debug=True:  loads one random image (fast, no cache).
      - debug=False: loads all images; uses a disk cache (./cache/images.pkl)
                     so repeated runs skip JPEG decoding and resizing.

    Args:
        path: Path to scan for images.
        cfg:  Configuration parameters.
    Returns:
        List of RGB images as numpy arrays.
    Raises:
        FileNotFoundError: if the directory does not exist.
    '''
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{path} is not a directory \n")

    image_paths = _collect_image_paths(path)
    if not image_paths:
        return []

    # -- Debug mode: load one random image, skip cache --
    if cfg.debug:
        chosen = image_paths[np.random.randint(0, len(image_paths))]
        img = _load_single(chosen, cfg)
        if img is None:
            raise ValueError(f"Could not load image: {chosen}")
        print(f"[debug] Loaded 1 image: {chosen}")
        return [img]

    # -- Full mode: use disk cache when available --
    if os.path.exists(_CACHE_FILE):
        cache_mtime  = os.path.getmtime(_CACHE_FILE)
        newest_image = max(os.path.getmtime(p) for p in image_paths)
        if cache_mtime >= newest_image:
            with open(_CACHE_FILE, "rb") as fh:
                images = pickle.load(fh)
            print(f"[cache] Loaded {len(images)} images from {_CACHE_FILE}")
            return images

    # -- Load from disk and write cache --
    images = []
    for img_path in image_paths:
        img = _load_single(img_path, cfg)
        if img is None:
            print(f" Warning: {img_path} skipped (empty or unreadable)")
            continue
        images.append(img)

    os.makedirs(os.path.dirname(_CACHE_FILE), exist_ok=True)
    with open(_CACHE_FILE, "wb") as fh:
        pickle.dump(images, fh)
    print(f"[cache] Saved {len(images)} images to {_CACHE_FILE}")

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



