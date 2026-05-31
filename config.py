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

    # -- Execution mode --------------------------------------------------------
    debug: bool = True

    # -- Image loading (load_images) -------------------------------------------
    image_dims: tuple[int, int] = (4284, 5712)

    # -- Patch sampling (get_multi_patches, get_avg_color) ---------------------
    patch_size: int = 10
    roi_background_color: tuple[int, int, int] = (30, 90, 170)

    # -- Color filtering (shared by get_ROI_from_color and binarize_image) -----
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

    # -- ROI detection (get_ROI_from_color) ------------------------------------
    roi_min_area_ratio: float = 0.03
    roi_padding: int = 30
    roi_open_kernel_dims: tuple[int, int] = (7, 7)
    roi_close_kernel_dims: tuple[int, int] = (21, 21)

    # -- CLAHE illumination (normalize_illumination_clahe) ---------------------
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: tuple[int, int] = (8, 8)

    # -- Binarization (binarize_image) -----------------------------------------
    # These morphological kernels are reused by both binarize_image (cleaning
    # the blue-background mask) and _binarize_tray (cleaning the Sauvola
    # foreground after dilation).
    bin_open_kernel_dims: tuple[int, int] = (3, 3)
    bin_close_kernel_dims: tuple[int, int] = (7, 7)
    bin_tolerance_h: float = 0.07
    bin_tolerance_s: float = 0.20
    bin_tolerance_v: float = 1.00

    # -- Tray crop (get_tray_crop) ---------------------------------------------
    tray_full_size_tol: float = 0.99
    tray_edge_close_kernel: int = 40

    # -- Specular reflection detection / repair --------------------------------
    reflection_v_threshold: int = 240
    reflection_s_threshold: int = 30
    reflection_inpaint_enabled: bool = True
    reflection_inpaint_radius: int = 15

    # -- Segmentation (segment_instruments) ------------------------------------
    seg_close_kernel_dims: tuple[int, int] = (16, 16)
    seg_sauvola_window_size: int = 51
    seg_sauvola_k: float = 0.2
    seg_min_contour_area: int = 500
    seg_median_area_threshold: float = 0.2

    # Outlier grading weights — must sum to 1.0.
    # Each weight controls how much the corresponding normalised score
    # contributes to the fused-instrument grade in _analyze_bbox_outliers:
    #   area_score: how much larger is this bbox relative to the smallest?
    #   fill_score: how much less compact (lower fill ratio) is this bbox?
    seg_outlier_weight_area: float = 0.6
    seg_outlier_weight_fill: float = 0.4

    # Grade threshold: a bbox is flagged as a candidate outlier when
    #   grade > seg_outlier_grade_threshold × median_grade
    seg_outlier_grade_threshold: float = 1.8

    # Secondary-pass ratio: among primary candidates, demote any whose grade
    # falls below  max_candidate_grade / seg_outlier_secondary_ratio.
    seg_outlier_secondary_ratio: float = 1.5

    # -- Watershed split (watershed.watershed_split) ---------------------------
    # Fraction of dist.max() above which a pixel is a definite foreground seed.
    # Lower → more pixels become seeds (risk of over-segmentation).
    ws_sure_fg_threshold: float = 0.3

    # MORPH_CLOSE kernel applied to the sure_fg mask to bridge fragments of the
    # SAME instrument (serrations, reflection gaps, ridge wobble).  Must stay
    # smaller than the inter-instrument gap (~50–200 px) so real contacts are
    # not merged.
    ws_seed_merge_kernel: tuple[int, int] = (15, 15)

    # Sure-background dilation: pushes the definite-background region outward
    # so the uncertain band covers the full contact zone.
    ws_sure_bg_dilate_kernel: tuple[int, int] = (5, 5)
    ws_sure_bg_dilate_iters: int = 2

    # cv.distanceTransform mask size: 3 = faster, 5 = more accurate.
    ws_dist_mask_size: int = 5

    # -- Concave point detection (Miró-Nicolau et al. 2024) -------------------
    # k-curvature look-around step (vertices on each side of the candidate).
    cp_k: int = 9
    # Minimum / maximum high-curvature run length for _find_rois_recursive.
    cp_lmin: int = 6
    cp_lmax: int = 25
    # Threshold increment at each recursion level.
    cp_delta_t: float = 0.1
    # RDP simplification epsilon (pixels).  Larger → fewer vertices.
    cp_rdp_epsilon: float = 2.0

    # -- Concave-point cut (separates fused instruments) ----------------------
    # Width of the black cut line drawn through each best concave point.
    # 3–7 px is enough to fully separate two instruments after the dilation
    # applied in _binarize_tray, while remaining narrow enough that the
    # resulting per-instrument bboxes keep an accurate footprint.
    concave_cut_line_width: int = 5
