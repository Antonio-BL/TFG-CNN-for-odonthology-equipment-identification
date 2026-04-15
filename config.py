from dataclasses import dataclass
from typing import Optional


@dataclass
class PreprocessConfig:

    # -- Execution mode --
    debug: bool = True

    # -- Image loading (load_images) --
    image_dims: tuple[int, int] = (4284, 5712)

    # -- Patch sampling (get_multi_patches, get_avg_color) --
    patch_size: int = 10
    patch_center: int = None
    ROI_background_color: tuple[int, int, int] = (30, 90, 170)

    # -- Color filtering (shared by get_ROI_from_color and binarize_image) --
    color_filter_method: str = "hsv"
    color_filter_tolerance_rgb: float = 0.5
    color_filter_tolerance_h: float = 0.10
    color_filter_tolerance_s: float = 0.30
    color_filter_tolerance_v: float = 1.00
    color_filter_hsv_limits: tuple[tuple[Optional[int], Optional[int]],
                                   tuple[Optional[int], Optional[int]],
                                   tuple[Optional[int], Optional[int]]] = (
        (None, None),
        (40,   220),
        (30,   220),
    )

    # -- ROI detection (get_ROI_from_color) --
    roi_min_area_ratio: float = 0.03
    roi_padding: int = 30
    roi_open_kernel_dims: tuple[int, int] = (7, 7)
    roi_close_kernel_dims: tuple[int, int] = (21, 21)

    # -- CLAHE illumination (normalize_illumination_clahe) --
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: tuple[int, int] = (8, 8)

    # -- Binarization (binarize_image) --
    open_kernel_dims: tuple[int, int] = (3, 3)
    close_kernel_dims: tuple[int, int] = (10, 10)

    # -- Tray crop (get_tray_crop) --
    tray_full_size_tol: float = 0.99

    # -- Specular reflection detection (detect_specular_reflections) --
    reflection_v_threshold: int = 240
    reflection_s_threshold: int = 30
    remove_reflections: bool = True

    # -- Segmentation (segment_instruments) --
    seg_close_kernel_dims: tuple[int, int] = (16, 16)
    seg_min_contour_area: int = 500
    seg_median_area_threshold: float = 0.2
