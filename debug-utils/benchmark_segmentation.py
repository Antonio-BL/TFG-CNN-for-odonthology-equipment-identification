"""benchmark_segmentation.py

Runs segmentation.py (with KNN clustering) and segmentation_test.py (without)
on every image in the Trays/ directory and writes a Markdown comparison report.
"""

import os
import sys
import traceback
import datetime
import numpy as np
import cv2 as cv

# Suppress any display-related side effects from the segmentation modules
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("DISPLAY", ":0")

from config     import PreprocessConfig
from preprocess import (get_ROI_from_color, binarize_image, get_tray_crop,
                        remove_blue_background, detect_specular_reflections)

import segmentation      as seg_knn
import segmentation_test as seg_base


TRAYS_DIR   = "./Trays"
REPORT_PATH = "./segmentation_benchmark_report.md"


def preprocess_image(image_path, cfg):
    img_bgr = cv.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load: {image_path}")
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    img_rgb = cv.resize(img_rgb, cfg.image_dims, interpolation=cv.INTER_AREA)

    roi_crop, _, _      = get_ROI_from_color(img_rgb, cfg)
    binary_mask         = binarize_image(roi_crop, cfg)
    tray_masked, _, _   = get_tray_crop(roi_crop, binary_mask, cfg)
    tray_no_bg          = remove_blue_background(tray_masked, cfg)
    return tray_no_bg


def run_benchmark():
    cfg = PreprocessConfig()

    image_paths = sorted([
        os.path.join(root, f)
        for root, _, files in os.walk(TRAYS_DIR)
        for f in files
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))
    ])

    if not image_paths:
        print(f"No images found in {TRAYS_DIR}")
        sys.exit(1)

    print(f"Found {len(image_paths)} images. Running benchmark...\n")

    rows = []
    for path in image_paths:
        name = os.path.basename(path)
        print(f"  Processing {name}...", end=" ", flush=True)

        try:
            tray_no_bg = preprocess_image(path, cfg)
        except Exception as e:
            print(f"PREPROCESS ERROR: {e}")
            rows.append({
                'name':         name,
                'error':        str(e),
                'knn_bboxes':   None,
                'base_bboxes':  None,
                'knn_outlier':  None,
                'base_outlier': None,
            })
            continue

        knn_n, knn_scenario   = None, None
        base_n, base_scenario = None, None
        knn_err = base_err = None

        try:
            _, knn_bboxes, knn_analysis = seg_knn.segment_instruments(tray_no_bg, cfg)
            knn_n        = len(knn_bboxes)
            knn_scenario = knn_analysis['scenario']
        except Exception as e:
            knn_err = str(e)

        try:
            _, base_bboxes, base_analysis = seg_base.segment_instruments(tray_no_bg, cfg)
            base_n        = len(base_bboxes)
            base_scenario = base_analysis['scenario']
        except Exception as e:
            base_err = str(e)

        print(f"KNN={knn_n}  base={base_n}")
        rows.append({
            'name':         name,
            'error':        None,
            'knn_bboxes':   knn_n,
            'base_bboxes':  base_n,
            'knn_outlier':  knn_scenario,
            'base_outlier': base_scenario,
            'knn_err':      knn_err,
            'base_err':     base_err,
        })

    _write_report(rows, image_paths)
    print(f"\nReport written to {REPORT_PATH}")


def _write_report(rows, image_paths):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    valid = [r for r in rows if r['knn_bboxes'] is not None and r['base_bboxes'] is not None]
    diffs = [r['knn_bboxes'] - r['base_bboxes'] for r in valid]

    lines = []
    lines.append("# Segmentation Benchmark Report")
    lines.append(f"\n**Generated:** {now}  ")
    lines.append(f"**Images processed:** {len(image_paths)}  ")
    lines.append(f"**Trays directory:** `{TRAYS_DIR}`\n")

    lines.append("## Overview\n")
    lines.append("Compares two segmentation pipelines on every image in the Trays dataset:\n")
    lines.append("| Pipeline | Description |")
    lines.append("|---|---|")
    lines.append("| **With KNN** (`segmentation.py`) | morphological close → Otsu → KNN/convex-hull clustering → bbox filter → outlier analysis |")
    lines.append("| **Without KNN** (`segmentation_test.py`) | morphological close → Otsu → bbox filter → outlier analysis |")

    if valid:
        avg_knn  = np.mean([r['knn_bboxes']  for r in valid])
        avg_base = np.mean([r['base_bboxes'] for r in valid])
        avg_diff = np.mean(diffs)
        lines.append("\n## Summary Statistics\n")
        lines.append("| Metric | With KNN | Without KNN | Δ (KNN − base) |")
        lines.append("|---|---|---|---|")
        lines.append(f"| Mean bboxes/image | {avg_knn:.1f} | {avg_base:.1f} | {avg_diff:+.1f} |")
        lines.append(f"| Min bboxes        | {min(r['knn_bboxes'] for r in valid)} | {min(r['base_bboxes'] for r in valid)} | — |")
        lines.append(f"| Max bboxes        | {max(r['knn_bboxes'] for r in valid)} | {max(r['base_bboxes'] for r in valid)} | — |")

        merged_count   = sum(1 for d in diffs if d < 0)
        split_count    = sum(1 for d in diffs if d > 0)
        unchanged      = sum(1 for d in diffs if d == 0)
        lines.append(f"| Images where KNN **reduced** bbox count (merging)  | — | — | {merged_count}/{len(valid)} |")
        lines.append(f"| Images where KNN **increased** bbox count (splitting) | — | — | {split_count}/{len(valid)} |")
        lines.append(f"| Images **unchanged** | — | — | {unchanged}/{len(valid)} |")

    lines.append("\n## Per-Image Results\n")
    lines.append("| Image | With KNN (bboxes) | Outlier scenario | Without KNN (bboxes) | Outlier scenario | Δ | Notes |")
    lines.append("|---|---|---|---|---|---|---|")

    for r in rows:
        name = r['name']
        if r.get('error'):
            lines.append(f"| `{name}` | — | — | — | — | — | ⚠ preprocess error: {r['error']} |")
            continue

        knn_n  = r['knn_bboxes']  if r['knn_bboxes']  is not None else "ERR"
        base_n = r['base_bboxes'] if r['base_bboxes'] is not None else "ERR"

        if isinstance(knn_n, int) and isinstance(base_n, int):
            delta = knn_n - base_n
            delta_str = f"{delta:+d}" if delta != 0 else "0"
            note = ""
            if delta < 0:
                note = "KNN merged fragments"
            elif delta > 0:
                note = "KNN split / added regions"
        else:
            delta_str = "—"
            note = f"knn_err={r.get('knn_err','')} base_err={r.get('base_err','')}"

        knn_out  = r['knn_outlier']  or "—"
        base_out = r['base_outlier'] or "—"

        lines.append(
            f"| `{name}` | {knn_n} | {knn_out} | {base_n} | {base_out} | {delta_str} | {note} |"
        )

    lines.append("\n## Interpretation Notes\n")
    lines.append(
        "- **Δ < 0**: KNN clustering merged contour fragments into fewer, larger bounding boxes "
        "(desirable when instrument parts were split across multiple contours).\n"
        "- **Δ > 0**: KNN clustering introduced extra bounding boxes, possibly by expanding convex "
        "hulls over unrelated regions.\n"
        "- **Δ = 0**: Both pipelines agree on the instrument count for this image.\n"
        "- **Outlier scenario `single`**: one bbox is ≥2× the median area — likely two touching "
        "instruments fused into one contour.\n"
        "- **Outlier scenario `multiple`**: two or more such fused regions detected.\n"
    )

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    run_benchmark()
