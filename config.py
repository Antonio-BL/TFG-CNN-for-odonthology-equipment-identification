from dataclasses import dataclass
from typing import Optional


@dataclass
class PreprocessConfig:
    """Central configuration for the segmentation pipeline.

    Naming convention: every tunable parameter is prefixed by the pipeline
    stage that owns it.

        general (no prefix) : runtime / IO / image dims
        patch_*             : patch sampling for background-color estimation
        color_filter_*      : shared HSV / RGB colour-mask parameters
        roi_*               : Step 1, ROI detection
        clahe_*             : CLAHE illumination normalisation
        bin_*               : Step 3, binarization of the blue background
        tray_*              : Step 4, tray crop
        reflection_*        : specular reflection detection / inpainting
        seg_*               : Step 5, instrument segmentation and outlier filter
        ws_*                : watershed split of fused instruments
    """

    # -- Execution mode -----------------------------------------------------
    debug: bool = True

    # -- Image loading (load_images) ----------------------------------------
    image_dims: tuple[int, int] = (4284, 5712)

    # -- Patch sampling (get_multi_patches, get_avg_color) ------------------
    patch_size: int = 10
    roi_background_color: tuple[int, int, int] = (30, 90, 170)

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

    # -- ROI detection (get_ROI_from_color) ---------------------------------
    roi_min_area_ratio: float = 0.03
    roi_padding: int = 30
    roi_open_kernel_dims: tuple[int, int] = (7, 7)
    roi_close_kernel_dims: tuple[int, int] = (21, 21)

    # -- CLAHE illumination (normalize_illumination_clahe) ------------------
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: tuple[int, int] = (8, 8)

    # -- Binarization (binarize_image) --------------------------------------
    # These morphological kernels are reused by both binarize_image (cleaning
    # the blue-background mask) and _binarize_tray (cleaning the Sauvola
    # foreground after dilation). Keeping them shared because both currently
    # benefit from the same (3, 3) / (7, 7) sizing.
    bin_open_kernel_dims: tuple[int, int] = (3, 3)
    bin_close_kernel_dims: tuple[int, int] = (7, 7)
    bin_tolerance_h: float = 0.07
    bin_tolerance_s: float = 0.20
    bin_tolerance_v: float = 1.00

    # -- Tray crop (get_tray_crop) ------------------------------------------
    tray_full_size_tol: float = 0.99
    tray_edge_close_kernel: int = 40

    # -- Specular reflection detection / repair -----------------------------
    reflection_v_threshold: int = 240
    reflection_s_threshold: int = 30
    reflection_inpaint_enabled: bool = True
    reflection_inpaint_radius: int = 15

    # -- Segmentation (segment_instruments) ---------------------------------
    seg_close_kernel_dims: tuple[int, int] = (16, 16)
    seg_sauvola_window_size: int = 51
    seg_sauvola_k: float = 0.2
    seg_min_contour_area: int = 500
    seg_median_area_threshold: float = 0.2
    # Outlier grading. Weights MUST sum to 1.0; they combine the per-bbox
    # area / edge / fill scores in _analyze_bbox_outliers.
    seg_outlier_weight_area: float = 0.50
    seg_outlier_weight_edge: float = 0.25
    seg_outlier_weight_fill: float = 0.25
    seg_outlier_grade_threshold: float = 1.8       # grade > X × median ⇒ candidate
    seg_outlier_secondary_ratio: float = 1.5       # demote any candidate below max/X
    # |Laplacian(seg_binary)| > this threshold marks an edge pixel.
    seg_edge_magnitude_threshold: float = 1.0
    # Cleanup kernel applied AFTER the watershed cut in split_fused_instruments
    # to remove single-pixel spike artefacts left along the incision line.
    seg_split_open_kernel_dims: tuple[int, int] = (3, 3)

    # -- Watershed (apply_watershed_to_outliers) ----------------------------
    # Fraction of dist.max() above which pixels are definite foreground seeds.
    ws_sure_fg_threshold: float = 0.3
    # MORPH_CLOSE kernel applied to sure_fg to bridge fragments of the SAME
    # instrument (serrations, reflection gaps, ridge wobble) before
    # connectedComponents. Must stay smaller than the gap between the
    # distance-ridge peaks of two touching instruments (~50–200 px) so that
    # real contacts are NOT merged.
    ws_seed_merge_kernel: tuple[int, int] = (15, 15)
    ws_sure_bg_dilate_kernel: tuple[int, int] = (5, 5)
    ws_sure_bg_dilate_iters: int = 2
    ws_dist_mask_size: int = 5                     # 3 = faster, 5 = more accurate
    ws_overlay_alpha: float = 0.55                 # opacity of coloured segment fill
    ws_heatmap_colormap: str = 'hot'
