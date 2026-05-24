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
    close_kernel_dims: tuple[int, int] = (7, 7)
    bin_tolerance_h: float = 0.07
    bin_tolerance_s: float = 0.20
    bin_tolerance_v: float = 1.00

    # -- Tray crop (get_tray_crop) --
    tray_full_size_tol: float = 0.99
    tray_edge_close_kernel: int = 40

    # -- Specular reflection detection / repair (detect_specular_reflections) --
    reflection_v_threshold: int = 240
    reflection_s_threshold: int = 30
    inpaint_reflections: bool = True
    reflection_inpaint_radius: int = 15

    # -- Segmentation (segment_instruments) --
    seg_close_kernel_dims: tuple[int, int] = (16, 16)
    sauvola_window_size: int = 51
    sauvola_k: float = 0.2
    seg_min_contour_area: int = 500
    seg_median_area_threshold: float = 0.2
    seg_outlier_grade_threshold: float = 1.8  # grade > XX × median grade ⇒ outlier candidate
    seg_outlier_secondary_ratio: float = 1.5  # among candidates, drop any with grade < max_candidate_grade / 1.5

    # -- Watershed (apply_watershed_to_outliers) --
    ws_sure_fg_threshold: float = 0.3          # fraction of dist.max() above which pixels are definite foreground seeds
    ws_seed_merge_kernel: tuple[int, int] = (15, 15)    # MORPH_CLOSE kernel applied to sure_fg before connectedComponents.
                                                         # Bridges seed fragments inside a single instrument (caused by
                                                         # serrations, reflection gaps, ridge wobble). Must be SMALLER than
                                                         # the typical gap between the distance-ridge peaks of two touching
                                                         # instruments — otherwise it will merge true instruments. (15, 15)
                                                         # is a good default for dental tools at the current image resolution.
    ws_sure_bg_dilate_kernel: tuple[int, int] = (5, 5)  # ellipse kernel for dilating the ROI to define sure background
    ws_sure_bg_dilate_iters: int = 2           # dilation iterations for sure-background mask
    ws_dist_mask_size: int = 5                 # distance-transform mask size (3 = faster, 5 = more accurate)
    ws_heatmap_colormap: str = 'hot'           # matplotlib colormap for the distance-transform heatmap panel
