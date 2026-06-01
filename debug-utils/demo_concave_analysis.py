# debug-utils/demo_concave_analysis.py
# Visual spread of all concave-point properties detected on one image.
#
# Purpose:
#   Study which property (kappa, roi_len, roi_kappa_mean, roi_kappa_max,
#   chord_len) best identifies the "true" segmentation point among all
#   candidates found on a fused-instrument blob.
#
# Layout per outlier bbox:
#   Left   — Cropped tray image of the outlier region with each concave
#             point marked as a numbered coloured circle.  The chosen
#             cut point (max chord_len) carries a gold star.
#   Centre — Property strip: one horizontal-bar subplot per property,
#             bars coloured to match the markers on the left panel.
#             The best-chord bar carries a gold border.
#   Right  — Compact numeric table for exact values.
#
# Usage:
#   cd /home/antonio/Documents/TFG-project
#   python debug-utils/demo_concave_analysis.py
#
# or override the image at the bottom of this file.

from __future__ import annotations

import os
import sys

# Allow importing project modules from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import to_rgba

from run_pipeline              import main as run_pipeline
from pipeline.concave_cut      import _concave_grade

# ── Palette ──────────────────────────────────────────────────────────────────
# Distinct colours for up to 12 concave points; beyond that the palette wraps.
_POINT_COLOURS = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
    '#dcbeff', '#9a6324',
]

_PROPERTIES = [
    ('chord_len',      'Chord length  [px]',       True),   # (key, label, higher=better?)
    ('kappa',          'κ  (k-curvature)',          True),
    ('roi_kappa_max',  'ROI κ max',                 True),
    ('roi_kappa_mean', 'ROI κ mean',                True),
    ('roi_len',        'ROI length  [vertices]',    False),  # neutral — not directional
]

# Margin (in pixels in the original image) added around the oriented bbox when
# cropping for the left panel.
_CROP_PAD_PX = 60


def _colour(idx: int) -> str:
    return _POINT_COLOURS[idx % len(_POINT_COLOURS)]


def _crop_bbox_region(
    img_rgb: np.ndarray,
    bbox: tuple,
    pad: int = _CROP_PAD_PX,
) -> tuple[np.ndarray, int, int]:
    """Return a padded axis-aligned crop around the oriented bbox.

    Returns (crop_rgb, x_offset, y_offset) where (x_offset, y_offset) is the
    top-left corner of the crop in the full image.
    """
    h_img, w_img = img_rgb.shape[:2]
    center, size, angle = bbox[0], bbox[1], bbox[2]
    box_pts = np.int32(cv.boxPoints((center, size, angle)))
    ax, ay, aw, ah = cv.boundingRect(box_pts)

    x1 = max(0, ax - pad)
    y1 = max(0, ay - pad)
    x2 = min(w_img, ax + aw + pad)
    y2 = min(h_img, ay + ah + pad)

    return img_rgb[y1:y2, x1:x2].copy(), x1, y1


def _draw_bbox_outline(ax_img, bbox: tuple, ox: int, oy: int) -> None:
    """Draw the oriented bbox as a dashed white polygon on the image axis."""
    center, size, angle = bbox[0], bbox[1], bbox[2]
    box_pts = np.int32(cv.boxPoints((center, size, angle)))
    # Shift to crop coordinates
    box_pts_shifted = box_pts - np.array([[ox, oy]])
    poly = mpatches.Polygon(
        box_pts_shifted, closed=True,
        linewidth=1.5, edgecolor='white', facecolor='none',
        linestyle='--', zorder=3,
    )
    ax_img.add_patch(poly)


def _plot_one_bbox(
    fig: plt.Figure,
    outer_gs,
    bbox_idx: int,
    bbox: tuple,
    pts: list[dict],
    tray_img: np.ndarray,
) -> None:
    """Fill one row of the figure for a single outlier bbox."""
    n_pts = len(pts)
    if n_pts == 0:
        return

    # Find the best point using the same weighted grade as concave_cut
    # grade = 0.20·chord_len + 0.50·kappa + 0.30·roi_kappa_mean (all normalised)
    best_idx = max(range(n_pts), key=lambda i: _concave_grade(pts[i], pts))

    # ── Inner grid: [image | bar charts per property | table] ───────────────
    n_props = len(_PROPERTIES)
    inner_gs = gridspec.GridSpecFromSubplotSpec(
        1, 3,
        subplot_spec=outer_gs,
        width_ratios=[2, n_props, 1.5],
        wspace=0.08,
    )

    # ── Left: image crop with numbered markers ───────────────────────────────
    crop, ox, oy = _crop_bbox_region(tray_img, bbox)
    ax_img = fig.add_subplot(inner_gs[0])
    ax_img.imshow(crop)
    _draw_bbox_outline(ax_img, bbox, ox, oy)

    for i, pt in enumerate(pts):
        cx = pt['x'] - ox
        cy = pt['y'] - oy
        col = _colour(i)
        # Outer black halo for contrast
        ax_img.plot(cx, cy, 'o', color='black', markersize=16, zorder=4)
        ax_img.plot(cx, cy, 'o', color=col,     markersize=13, zorder=5)
        ax_img.text(
            cx, cy, str(i + 1),
            ha='center', va='center',
            fontsize=7, fontweight='bold', color='white', zorder=6,
        )
        if i == best_idx:
            ax_img.plot(cx, cy - 14, '*',
                        color='gold', markersize=11, markeredgecolor='black',
                        markeredgewidth=0.7, zorder=7)

    ax_img.set_title(f'Outlier bbox #{bbox_idx}  ({n_pts} concave pts)',
                     fontsize=9, fontweight='bold', pad=4)
    ax_img.axis('off')

    # ── Centre: one horizontal-bar subplot per property ──────────────────────
    # Sub-divide the centre cell into n_props rows
    props_gs = gridspec.GridSpecFromSubplotSpec(
        n_props, 1,
        subplot_spec=inner_gs[1],
        hspace=0.55,
    )

    y_positions = np.arange(n_pts)
    bar_labels  = [str(i + 1) for i in range(n_pts)]

    for prop_row, (key, label, _higher_is_better) in enumerate(_PROPERTIES):
        ax_bar = fig.add_subplot(props_gs[prop_row])

        values = np.array([pt[key] for pt in pts], dtype=float)
        colours = [_colour(i) for i in range(n_pts)]

        # Edge colours: gold border for the best point
        edge_colours = ['gold' if i == best_idx else 'none' for i in range(n_pts)]
        edge_widths  = [2.0   if i == best_idx else 0.0   for i in range(n_pts)]

        bars = ax_bar.barh(
            y_positions, values,
            color=colours, edgecolor=edge_colours, linewidth=edge_widths,
            height=0.7,
        )

        # Value labels inside bars (or just outside if bar is very thin)
        x_max = values.max() if values.max() > 0 else 1.0
        for j, (bar, val) in enumerate(zip(bars, values)):
            text_x = val + x_max * 0.02
            ax_bar.text(
                text_x, bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}',
                va='center', ha='left',
                fontsize=6.5, color='#333333',
            )

        ax_bar.set_yticks(y_positions)
        ax_bar.set_yticklabels(bar_labels, fontsize=7)
        ax_bar.set_xlabel(label, fontsize=7, labelpad=2)
        ax_bar.tick_params(axis='x', labelsize=6)
        ax_bar.set_xlim(0, x_max * 1.35)
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)

        # Shade background of the best-point bar row
        ax_bar.axhspan(
            best_idx - 0.5, best_idx + 0.5,
            facecolor='gold', alpha=0.10, zorder=0,
        )

    # ── Right: compact numeric table ─────────────────────────────────────────
    ax_tbl = fig.add_subplot(inner_gs[2])
    ax_tbl.axis('off')

    col_labels = ['#', 'chord', 'κ', 'κmax', 'κmean', 'roi_l']
    tbl_data = []
    for i, pt in enumerate(pts):
        row = [
            str(i + 1),
            f'{pt["chord_len"]:.1f}',
            f'{pt["kappa"]:.3f}',
            f'{pt["roi_kappa_max"]:.3f}',
            f'{pt["roi_kappa_mean"]:.3f}',
            str(pt['roi_len']),
        ]
        tbl_data.append(row)

    table = ax_tbl.table(
        cellText=tbl_data,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.3)

    # Colour the # column cells and highlight best row
    for r in range(n_pts):
        # Point colour on the # cell
        table[r + 1, 0].set_facecolor(to_rgba(_colour(r), 0.85))
        table[r + 1, 0].set_text_props(color='white', fontweight='bold')
        if r == best_idx:
            for c in range(1, len(col_labels)):
                table[r + 1, c].set_facecolor(to_rgba('gold', 0.25))
                table[r + 1, c].set_text_props(fontweight='bold')

    # Header cells
    for c in range(len(col_labels)):
        table[0, c].set_facecolor('#333333')
        table[0, c].set_text_props(color='white', fontweight='bold')

    ax_tbl.set_title('Exact values', fontsize=8, pad=4)

    # Legend note about gold highlight
    fig.text(
        0.5, 0.01,
        '★ gold  =  chosen cut point  (grade = 0.20·chord_len + 0.50·κ + 0.30·roi_κ_mean, all min-max normalised)',
        ha='center', va='bottom', fontsize=8, style='italic', color='#555555',
    )


def plot_concave_analysis(
    concave_pts: dict[int, list[dict]],
    outlier_bboxes: list[tuple],
    tray_img: np.ndarray,
    image_label: str | None = None,
) -> None:
    """Produce one figure per outlier bbox with a full property spread.

    Args:
        concave_pts:    dict[bbox_idx → list of point dicts] from
                        detect_concave_points.
        outlier_bboxes: List of (center, size, angle, area) tuples, same
                        indexing as concave_pts.
        tray_img:       RGB image (e.g., tray_no_bg) used as backdrop.
        image_label:    Optional string for the window title.
    """
    active = [
        (idx, bbox)
        for idx, bbox in enumerate(outlier_bboxes)
        if concave_pts.get(idx)
    ]

    if not active:
        print('[demo_concave_analysis] No concave points detected — nothing to plot.')
        return

    for bbox_idx, bbox in active:
        pts = concave_pts[bbox_idx]
        n_pts = len(pts)

        title = f'Concave-point analysis — bbox #{bbox_idx}  ({n_pts} pts)'
        if image_label:
            title = f'{image_label}  ·  {title}'

        fig = plt.figure(figsize=(20, max(5, 2.5 * len(_PROPERTIES))))
        fig.suptitle(title, fontsize=12, fontweight='bold', y=0.98)

        outer_gs = gridspec.GridSpec(1, 1, figure=fig,
                                     top=0.93, bottom=0.07,
                                     left=0.03, right=0.97)

        _plot_one_bbox(fig, outer_gs[0], bbox_idx, bbox, pts, tray_img)

        plt.show()
        plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

def _process_one(result: tuple, label: str) -> None:
    """Unpack a pipeline result tuple and call plot_concave_analysis."""
    tray_no_bg, _seg_binary, _seg_cut, _final_bboxes, outlier_analysis, concave_pts = result

    outlier_bboxes = [e['bbox'] for e in outlier_analysis['outliers']]

    if not outlier_bboxes:
        print(f'[demo_concave_analysis] {label}: no outlier bboxes — skipping.')
        return

    total_pts = sum(len(v) for v in concave_pts.values())
    print(f'\n[demo_concave_analysis] {label}: '
          f'{len(outlier_bboxes)} outlier bbox(es), '
          f'{total_pts} concave point(s) total.')

    plot_concave_analysis(concave_pts, outlier_bboxes, tray_no_bg,
                          image_label=label)


if __name__ == '__main__':
    # None   → one random image from ./Trays
    # 'all'  → every image in ./Trays  (figures appear one by one; close each to advance)
    # or a specific path, e.g. './Trays/IMG_3353.jpg'
    IMAGE_PATH: str | None = None
    IMAGE_PATH = 'all'
    if IMAGE_PATH == 'all':
        results = run_pipeline(debugging=False, image_path='all')
        # run_pipeline('all') returns a list of 6-tuples, one per image.
        # Reconstruct the sorted path list to derive a label for each result.
        tray_paths = sorted(
            os.path.join(wd, f)
            for wd, _, files in os.walk(
                os.path.join(os.path.dirname(__file__), '..', 'Trays')
            )
            for f in files
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
        )
        for path, res in zip(tray_paths, results):
            _process_one(res, os.path.basename(path))
    else:
        result = run_pipeline(debugging=False, image_path=IMAGE_PATH)
        label  = os.path.basename(IMAGE_PATH) if IMAGE_PATH else 'random image'
        _process_one(result, label)
