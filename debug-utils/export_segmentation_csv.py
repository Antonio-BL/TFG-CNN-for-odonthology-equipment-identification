"""export_segmentation_csv.py

Runs the full segmentation pipeline on every image in Trays/ (sorted
alphabetically) and writes one CSV row per bounding box.

Columns
-------
Image identity   : image_name, bbox_index, total_bboxes
Geometry         : center_x/y, width, height, angle_deg, long_side, short_side
Area / fill      : contour_area, bbox_area, perimeter, fill_ratio, positive_pixels
Ratios to median : area_ratio, side_ratio  (how far each bbox is from the median)
Cutoffs          : area_cutoff, side_cutoff  (actual threshold values used)
Outlier result   : is_outlier, scenario, median_contour_area, median_long_side
Config used      : one column per relevant PreprocessConfig field
"""

import csv
import glob
import os

import cv2 as cv
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("DISPLAY", ":0")

import pathlib as _pathlib
import sys as _sys
_ROOT = _pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from config      import PreprocessConfig
from pipeline.preprocess  import (get_ROI_from_color, binarize_image, get_tray_crop,
                                  remove_blue_background)
from pipeline.segmentation import segment_instruments, _compute_bbox_stats

TRAYS_DIR  = "./Trays"
OUTPUT_CSV = "./segmentation_results.csv"

FIELDNAMES = [
    # --- Image identity ---
    "image_name",
    "bbox_index",
    "total_bboxes",
    # --- Bounding box geometry ---
    "center_x",
    "center_y",
    "width",
    "height",
    "angle_deg",
    "long_side",
    "short_side",
    # --- Area / fill metrics ---
    "contour_area",
    "bbox_area",
    "perimeter",
    "fill_ratio",
    "positive_pixels",
    # --- Distance to median (how anomalous is this bbox) ---
    "area_ratio",        # contour_area / median_contour_area
    "side_ratio",        # long_side    / median_long_side
    # --- Absolute cutoff values for this image ---
    "area_cutoff",
    "side_cutoff",
    # --- Outlier classification ---
    "is_outlier",
    "scenario",
    "median_contour_area",
    "median_long_side",
    # --- Config parameters used for the analysis ---
    "cfg_seg_close_kernel_dims",
    "cfg_seg_min_contour_area",
    "cfg_seg_median_area_threshold",
    "cfg_seg_outlier_area_ratio",
    "cfg_seg_outlier_side_ratio",
]


def _preprocess(path: str, cfg: PreprocessConfig):
    img_bgr = cv.imread(path)
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    img_rgb = cv.resize(img_rgb, cfg.image_dims, interpolation=cv.INTER_AREA)

    roi_crop, _, _    = get_ROI_from_color(img_rgb, cfg)
    binary_mask       = binarize_image(roi_crop, cfg)
    tray_masked, _, _ = get_tray_crop(roi_crop, binary_mask, cfg)
    tray_no_bg        = remove_blue_background(tray_masked, cfg)
    return tray_no_bg


def main():
    cfg = PreprocessConfig()

    image_paths = sorted(glob.glob(os.path.join(TRAYS_DIR, "*.jpg")))
    if not image_paths:
        raise FileNotFoundError(f"No .jpg files found in {TRAYS_DIR}")

    print(f"Found {len(image_paths)} images. Running pipeline...\n")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for path in image_paths:
            fname = os.path.basename(path)
            print(f"  {fname}", end=" ... ", flush=True)

            try:
                tray_no_bg = _preprocess(path, cfg)
            except Exception as e:
                print(f"PREPROCESS ERROR: {e}")
                continue

            seg_binary, bboxes, oa, _ = segment_instruments(tray_no_bg, cfg)

            outlier_ids      = {id(e["bbox"]) for e in oa["outliers"]}
            total            = len(bboxes)
            median_area      = oa["median_area"]
            median_long_side = oa["median_long_side"]
            area_cutoff      = cfg.seg_outlier_area_ratio * median_area
            side_cutoff      = cfg.seg_outlier_side_ratio * median_long_side

            for i, bbox in enumerate(bboxes):
                center, size, angle, ca = bbox
                cx, cy = center
                w, h   = size
                ls     = max(w, h)
                ss     = min(w, h)

                stats = _compute_bbox_stats(seg_binary, bbox)

                writer.writerow({
                    "image_name":   fname,
                    "bbox_index":   i + 1,
                    "total_bboxes": total,
                    "center_x":     round(cx, 1),
                    "center_y":     round(cy, 1),
                    "width":        round(w, 1),
                    "height":       round(h, 1),
                    "angle_deg":    round(angle, 2),
                    "long_side":    round(ls, 1),
                    "short_side":   round(ss, 1),
                    "contour_area": ca,
                    "bbox_area":    int(stats["bbox_area"]),
                    "perimeter":    round(stats["perimeter"], 1),
                    "fill_ratio":   round(stats["fill_ratio"], 4),
                    "positive_pixels": stats["positive_pixels"],
                    "area_ratio":   round(ca / median_area, 3)      if median_area      > 0 else "",
                    "side_ratio":   round(ls / median_long_side, 3) if median_long_side > 0 else "",
                    "area_cutoff":  round(area_cutoff, 1),
                    "side_cutoff":  round(side_cutoff, 1),
                    "is_outlier":   id(bbox) in outlier_ids,
                    "scenario":     oa["scenario"],
                    "median_contour_area": round(median_area, 1),
                    "median_long_side":    round(median_long_side, 1),
                    "cfg_seg_close_kernel_dims":       str(cfg.seg_close_kernel_dims),
                    "cfg_seg_min_contour_area":        cfg.seg_min_contour_area,
                    "cfg_seg_median_area_threshold":   cfg.seg_median_area_threshold,
                    "cfg_seg_outlier_area_ratio":      cfg.seg_outlier_area_ratio,
                    "cfg_seg_outlier_side_ratio":      cfg.seg_outlier_side_ratio,
                })

            print(f"{total} bboxes  scenario={oa['scenario']}")

    print(f"\nDone. Results written to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
