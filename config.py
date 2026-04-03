from dataclasses import dataclass
from typing import Optional

@dataclass
@dataclass
class PreprocessConfig:
    # General
    image_dims: tuple[int, int] = (4284, 5712)                              # Image dimensions during preprocessing (resolution)
    open_kernel_dims: tuple[int, int] = (3, 3)                              # Dimensions of the kernel for morphological open
    close_kernel_dims: tuple[int, int] = (20, 20)                           # Dimensions of the kernel for morphological close

    # Filtering parameters for ALL color filtering actions
    color_filter_method: str = "hsv"                                        # Color system used in the filtering: rgb | hsv
    color_filter_tolerance_rgb: float = 0.5                                 # Tolerance for the filtering using rgb colors
    color_filter_tolerance_h: float = 0.10                                  # HSV hue tolerance   — tight: only match background hue
    color_filter_tolerance_s: float = 0.30                                  # HSV sat tolerance   — medium
    color_filter_tolerance_v: float = 1.00                                  # HSV value tolerance — full range: shadows shift V, not H
    color_filter_hsv_limits: tuple[tuple[Optional[int], Optional[int]],
                                   tuple[Optional[int], Optional[int]],
                                   tuple[Optional[int], Optional[int]]] = (
        (None, None),   # H — no restriction
        (40,   220),    # S — excludes gray metal (S<40) and near-white (S>220)
        (30,   220)     # V — excludes very dark shadows and specular highlights
    )

    # Patch dimensions
    patch_center: int = None                                                # Center around which to get a patch of the image
    patch_size: int = 10                                                    # Length of the sides of the patch (% of shortest image dimension)

    # ROI detection parameters
    ROI_background_color: tuple[int, int, int] = (30, 90, 170)             # Fallback color of the background in RGB (usually blue), used if get_avg_color fails
    roi_padding: int = 30                                                   # Padding around clusters
    roi_min_area_ratio: float = 0.03                                        # Minimum valid cluster size as a percentage of the image area (default 3% of the total image area)
    roi_open_kernel_dims: tuple[int, int] = (7, 7)                         # Dimensions of the kernel for morphological open after ROI
    roi_close_kernel_dims: tuple[int, int] = (21, 21)                      # Dimensions of the kernel for morphological close after ROI

    # CLAHE illumination normalisation (used by binarize_image)
    clahe_clip_limit: float = 2.0                                            # Contrast limit for CLAHE; higher = more aggressive equalisation
    clahe_tile_grid: tuple[int, int] = (8, 8)                               # Grid size for CLAHE tile regions

    # Execution mode
    debug: bool = True                                                       # Debug mode: loads only one random image instead of the full dataset