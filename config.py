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
    color_filter_tolerance_hsv: float = 0.22                                # Tolerance for the filtering using hsv colors
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