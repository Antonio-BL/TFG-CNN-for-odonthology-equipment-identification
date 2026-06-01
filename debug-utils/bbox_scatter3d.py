#!/usr/bin/env python3
"""bbox_scatter3d.py

Load a fixed set of tray images, run the full preprocess + segmentation
pipeline on each, extract per-bbox metrics, and produce one 3-D scatter plot
per image to study fused/anomalous detections visually.

Run from the project root:
    python debug-utils/bbox_scatter3d.py
"""

import os
import sys
import pathlib

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers 3-D projection

# ---------------------------------------------------------------------------
# Resolve project root so we can import from segmentation.py / preprocess.py
# ---------------------------------------------------------------------------
_HERE = pathlib.Path(__file__).resolve().parent   # debug-utils/
_ROOT = _HERE.parent                              # project root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2 as cv  # noqa: E402

from config import PreprocessConfig  # noqa: E402
from pipeline.preprocess import (  # noqa: E402
    get_ROI_from_color,
    binarize_image,
    get_tray_crop,
    remove_blue_background,
)
from pipeline.segmentation import segment_instruments  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRAY_DIR = _ROOT / "Trays"
TARGET_IMAGES = ["IMG_3347.jpg", "IMG_3355.jpg", "IMG_3376.jpg"]
OUTPUT_DIR = _HERE / "output"


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_bbox_metrics(bboxes: list) -> list[dict]:
    """Compute the three scatter-axis metrics for every oriented bounding box.

    Args:
        bboxes: List of (center, (w, h), angle, contour_area) tuples as
                returned by segment_instruments.

    Returns:
        List of dicts (one per bbox), each containing:
          aspect_ratio – max(w, h) / min(w, h)  (elongation; ≥ 1.0)
          bbox_area    – w × h  (px²)
          fill_ratio   – (contour_area / bbox_area) × 100  (0–100 %)
    """
    metrics: list[dict] = []
    for _center, size, _angle, contour_area in bboxes:
        w, h = size
        long_side = max(w, h)
        short_side = min(w, h)
        aspect_ratio = long_side / short_side if short_side > 0 else float("inf")
        bbox_area = float(w * h)
        fill_ratio = (contour_area / bbox_area * 100.0) if bbox_area > 0 else 0.0
        metrics.append(
            {
                "aspect_ratio": aspect_ratio,
                "bbox_area": bbox_area,
                "fill_ratio": fill_ratio,
            }
        )
    return metrics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_scatter3d(
    image_name: str,
    metrics: list[dict],
    outlier_indices: set[int],
    output_path: pathlib.Path,
) -> None:
    """Render and save a 3-D scatter plot of bbox metrics for one image.

    Normal instruments are plotted in blue; outliers in red.  Each point is
    annotated with its instrument number (#1, #2 …).  A stats text box is
    placed in the bottom-left corner of the figure.

    Args:
        image_name:      Filename used as the plot title.
        metrics:         Output of compute_bbox_metrics.
        outlier_indices: Set of 0-based indices flagged as outliers.
        output_path:     Destination PNG path (parent dir is created if needed).
    """
    if not metrics:
        print(f"  [skip] {image_name}: no bboxes to plot.")
        return

    xs = [m["aspect_ratio"] for m in metrics]
    ys = [m["bbox_area"] for m in metrics]
    zs = [m["fill_ratio"] for m in metrics]
    colors = [
        "red" if i in outlier_indices else "royalblue"
        for i in range(len(metrics))
    ]

    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        xs, ys, zs,
        c=colors, s=90, depthshade=True,
        edgecolors="k", linewidths=0.4,
    )

    for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
        ax.text(x, y, z, f"  #{i + 1}", fontsize=8, color=colors[i], fontweight="bold")

    ax.set_xlabel("Aspect Ratio (long/short)", labelpad=8)
    ax.set_ylabel("BBox Area (px²)", labelpad=8)
    ax.set_zlabel("Fill Ratio (%)", labelpad=8)
    ax.set_title(
        f"{image_name}  –  {len(metrics)} instrument(s)",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )

    # Per-instrument stats text box at the bottom-left of the figure
    header = (
        f"{'#':>2}  {'aspect_ratio':>12}  {'bbox_area':>12}  "
        f"{'fill_ratio':>10}  label"
    )
    separator = "─" * len(header)
    rows = [header, separator]
    for i, m in enumerate(metrics):
        label = "outlier" if i in outlier_indices else "normal"
        rows.append(
            f"{i + 1:>2}  {m['aspect_ratio']:>12.3f}  "
            f"{m['bbox_area']:>12.0f}  {m['fill_ratio']:>10.2f}  {label}"
        )
    fig.text(
        0.01, 0.01,
        "\n".join(rows),
        ha="left", va="bottom",
        fontsize=7, family="monospace",
        bbox=dict(
            facecolor="lightyellow", alpha=0.85,
            edgecolor="gray", boxstyle="round,pad=0.4",
        ),
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=120, bbox_inches="tight")
    print(f"  Saved → {output_path}")
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def _run_pipeline(
    image_path: pathlib.Path,
    cfg: PreprocessConfig,
) -> tuple[np.ndarray, list, dict]:
    """Load one image and run the full preprocess + segmentation pipeline.

    Returns the binary mask, the complete bbox list that _analyze_bbox_outliers
    received (after median-area filtering, before outlier classification), and
    the outlier-analysis dict.

    Args:
        image_path: Absolute path to the source JPEG.
        cfg:        PreprocessConfig instance.

    Returns:
        seg_binary:       uint8 binary mask (H, W).
        bboxes:           All oriented bboxes as (center, size, angle, area).
        outlier_analysis: Dict returned by _analyze_bbox_outliers inside
                          segment_instruments.
    """
    img_bgr = cv.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load: {image_path}")

    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    img_rgb = cv.resize(img_rgb, cfg.image_dims, interpolation=cv.INTER_AREA)

    roi_crop, _, _ = get_ROI_from_color(img_rgb, cfg)
    binary_mask = binarize_image(roi_crop, cfg)
    tray_masked, _, _ = get_tray_crop(roi_crop, binary_mask, cfg)
    tray_no_bg = remove_blue_background(tray_masked, cfg)

    seg_binary, bboxes, outlier_analysis, _ = segment_instruments(tray_no_bg, cfg)
    return seg_binary, bboxes, outlier_analysis


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(
    image_name: str,
    metrics: list[dict],
    outlier_indices: set[int],
) -> None:
    """Print a per-instrument stats table to stdout.

    Args:
        image_name:      Filename shown in the table header.
        metrics:         Output of compute_bbox_metrics.
        outlier_indices: Set of 0-based indices flagged as outliers.
    """
    rule = "─" * 74
    print(f"\n{rule}")
    print(f"  {image_name}  –  {len(metrics)} instrument(s)")
    print(rule)
    print(
        f"  {'#':>2}  {'aspect_ratio':>12}  {'bbox_area (px²)':>16}  "
        f"{'fill_ratio (%)':>14}  label"
    )
    print(f"  {'─'*2}  {'─'*12}  {'─'*16}  {'─'*14}  {'─'*7}")
    for i, m in enumerate(metrics):
        label = "outlier" if i in outlier_indices else "normal"
        print(
            f"  {i + 1:>2}  {m['aspect_ratio']:>12.3f}  "
            f"{m['bbox_area']:>16.0f}  {m['fill_ratio']:>14.2f}  {label}"
        )
    print(rule)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Iterate over target images, run pipeline, and plot 3-D scatter per image."""
    cfg = PreprocessConfig()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for filename in TARGET_IMAGES:
        image_path = TRAY_DIR / filename
        print(f"\n[INFO] Processing {filename} …")

        if not image_path.exists():
            print(f"  [WARN] Not found: {image_path}  — skipping.")
            continue

        seg_binary, bboxes, outlier_analysis = _run_pipeline(image_path, cfg)

        # Map outlier bbox objects back to their index in the original list
        outlier_ids = {id(e["bbox"]) for e in outlier_analysis["outliers"]}
        outlier_indices = {i for i, b in enumerate(bboxes) if id(b) in outlier_ids}

        print(
            f"  {len(bboxes)} bbox(es) | outlier scenario: "
            f"{outlier_analysis['scenario']} | outliers: {sorted(outlier_indices)}"
        )

        metrics = compute_bbox_metrics(bboxes)
        stem = pathlib.Path(filename).stem
        out_png = OUTPUT_DIR / f"{stem}_scatter3d.png"

        _print_summary(filename, metrics, outlier_indices)
        plot_scatter3d(filename, metrics, outlier_indices, out_png)

    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()
