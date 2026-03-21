from dataclasses import dataclass
from typing import Optional

@dataclass
class PreprocessConfig:
    image_dims: tuple[int, int] = (4284, 5712)
    open_kernel_dims: tuple[int, int] = (3, 3)
    close_kernel_dims: tuple[int, int] = (20, 20)

    color_filter_method: str = "hsv"
    color_filter_tolerance_rgb: float = 0.5
    color_filter_tolerance_hsv: float = 0.22
    color_filter_hsv_limits: tuple = (
        (None, None),   # H
        (40,   220),    # S 
        (30,   220)     # V
    )

    patch_center: int = None
    patch_size: int = 10

    ROI_background_color: tuple[int, int, int] = (30, 90, 170)
    roi_padding: int = 30
    roi_min_area_ratio: float = 0.03
    roi_open_kernel_dims: tuple[int, int] = (7, 7)
    roi_close_kernel_dims: tuple[int, int] = (21, 21)