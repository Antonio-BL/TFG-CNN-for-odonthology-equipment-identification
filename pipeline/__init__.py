"""Core CV segmentation pipeline."""
from pipeline.segmentation import segment_instruments, find_bboxes
from pipeline.preprocess import (
    get_ROI_from_color, binarize_image, get_tray_crop,
    remove_blue_background, detect_specular_reflections,
)
from pipeline.concave_points import detect_concave_points
from pipeline.concave_cut import (
    select_best_concave_points, apply_concave_cuts, _concave_grade,
)
