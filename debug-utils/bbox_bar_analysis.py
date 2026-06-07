# debug-utils/bbox_bar_analysis.py
# Visual spread of all bounding-box properties detected on one image.
#
# Purpose:
#   Study which properties (area_score, fill_score, grade, aspect_ratio,
#   bbox_area, fill_ratio) distinguish fused/outlier bboxes from normal ones.
#   Mirror format of demo_concave_analysis.py: one figure per image, three
#   columns — tray image with all bboxes labeled | bar charts per property |
#   compact numeric table.
#
# Layout per image:
#   Left   — Full tray image with every bbox drawn and numbered.
#             Each bbox CONTOUR uses that bbox's palette colour, matching its
#             bars and table row.  Outlier (fused-candidate) bboxes are drawn
#             with a thicker DASHED contour; normal bboxes are solid.
#             A numbered circle is kept as a fallback when colours look alike.
#   Centre — Property strip: one horizontal-bar subplot per property, bars
#             coloured to match the numbered markers.  Outlier bars carry a
#             gold border and a gold background band.
#   Right  — Compact numeric table for exact values.
#
# Usage:
#   cd /home/antonio/Documents/TFG-project
#   python debug-utils/bbox_bar_analysis.py
#
# Override IMAGE_PATH at the bottom:
#   None   → one random image from ./Trays
#   'all'  → every image in ./Trays (close each figure to advance)
#   or a specific path, e.g. './Trays/IMG_3353.jpg'

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import to_rgba

from run_pipeline import main as run_pipeline

# ── Palette (same as demo_concave_analysis) ───────────────────────────────────
_ITEM_COLOURS = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
    '#dcbeff', '#9a6324',
]

# Properties to visualise.  (key, label, higher_means_outlier?)
# higher_means_outlier = True  → a taller bar is more suspicious
_PROPERTIES = [
    ('grade',        'Grade  (outlier score)',         True),
    ('area_score',   'Area score',                     True),
    ('fill_score',   'Fill score',                     True),
    ('aspect_ratio', 'Aspect ratio  (long / short)',   True),
    ('bbox_area',    'BBox area  [px²]',               True),
    ('fill_ratio',   'Fill ratio  [%]',                False),  # lower = less compact
]

# Extra padding around the tray image (fraction of figure) — kept small because
# we show the full image, not a crop.
_IMG_DOWNSCALE = 4   # show tray at 1/N of original size for readability


def _colour(idx: int) -> str:
    return _ITEM_COLOURS[idx % len(_ITEM_COLOURS)]


def _compute_extra_metrics(bboxes: list[tuple]) -> list[dict]:
    """Derive aspect_ratio, bbox_area, fill_ratio from raw bbox tuples.

    These mirror compute_bbox_metrics from bbox_scatter3d.py.

    Args:
        bboxes: list of (center, (w, h), angle, contour_area) tuples.

    Returns:
        list of dicts with keys aspect_ratio, bbox_area, fill_ratio.
    """
    out = []
    for _center, size, _angle, contour_area in bboxes:
        w, h = size
        long_side  = max(w, h)
        short_side = min(w, h)
        aspect_ratio = long_side / short_side if short_side > 0 else float('inf')
        bbox_area    = float(w * h)
        fill_ratio   = (contour_area / bbox_area * 100.0) if bbox_area > 0 else 0.0
        out.append({
            'aspect_ratio': aspect_ratio,
            'bbox_area':    bbox_area,
            'fill_ratio':   fill_ratio,
        })
    return out


def _build_per_bbox_records(
    bboxes: list[tuple],
    outlier_analysis: dict,
) -> tuple[list[dict], set[int]]:
    """Merge outlier-analysis scores with geometric metrics into one record per bbox.

    Returns:
        records:         list[dict] — one per bbox, same order as `bboxes`.
        outlier_indices: set of 0-based indices classified as outliers.
    """
    # Build id → entry lookup from both sub-lists
    score_lookup: dict[int, dict] = {}
    for entry in outlier_analysis.get('outliers', []):
        score_lookup[id(entry['bbox'])] = entry
    for entry in outlier_analysis.get('normal', []):
        score_lookup[id(entry['bbox'])] = entry

    outlier_ids     = {id(e['bbox']) for e in outlier_analysis.get('outliers', [])}
    outlier_indices = {i for i, b in enumerate(bboxes) if id(b) in outlier_ids}

    extra = _compute_extra_metrics(bboxes)
    nan   = float('nan')

    records = []
    for i, (bbox, geom) in enumerate(zip(bboxes, extra)):
        entry = score_lookup.get(id(bbox), {})
        records.append({
            'grade':        entry.get('grade',      nan),
            'area_score':   entry.get('area_score', nan),
            'fill_score':   entry.get('fill_score', nan),
            'aspect_ratio': geom['aspect_ratio'],
            'bbox_area':    geom['bbox_area'],
            'fill_ratio':   geom['fill_ratio'],
        })

    return records, outlier_indices


def _draw_dashed_polygon(
    img: np.ndarray, pts: np.ndarray, colour_bgr: tuple,
    thickness: int, dash: int = 34, gap: int = 26,
) -> None:
    """Draw a closed polygon with a dashed edge style (OpenCV has no native dash)."""
    pts = np.asarray(pts, dtype=np.float32)
    n = len(pts)
    for i in range(n):
        p0, p1 = pts[i], pts[(i + 1) % n]
        seg = p1 - p0
        length = float(np.hypot(seg[0], seg[1]))
        if length < 1.0:
            continue
        unit = seg / length
        d = 0.0
        while d < length:
            a = p0 + unit * d
            b = p0 + unit * min(d + dash, length)
            cv.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                    colour_bgr, thickness, cv.LINE_AA)
            d += dash + gap


def _draw_tray_with_bboxes(
    tray_img: np.ndarray,
    bboxes: list[tuple],
    outlier_indices: set[int],
) -> np.ndarray:
    """Return an RGB image with every bbox drawn and its index labelled.

    Each bbox CONTOUR is drawn in that bbox's palette colour — the same colour as
    its bars in the centre panel and its row in the table — so a box maps to its
    bars at a glance.  Outlier (fused-candidate) bboxes use a thicker DASHED
    contour to stand out; normal bboxes use a solid contour.  The numbered circle
    is kept as a fallback for when two palette colours look similar.
    """
    vis = tray_img.copy()

    for i, bbox in enumerate(bboxes):
        center, size, angle = bbox[0], bbox[1], bbox[2]
        box_pts = np.int32(cv.boxPoints((center, size, angle)))
        is_outlier = i in outlier_indices

        # Contour colour = this bbox's palette colour (matches its bars / table row).
        # vis is an RGB image (drawn via imshow), so colours must be passed in RGB
        # order — NOT OpenCV's usual BGR, or red/blue end up swapped vs the bars.
        col_hex = _colour(i)
        r = int(col_hex[1:3], 16)
        g = int(col_hex[3:5], 16)
        b = int(col_hex[5:7], 16)
        colour_rgb = (r, g, b)

        if is_outlier:
            # Outlier candidate: thick + dashed so it is unmistakable.
            _draw_dashed_polygon(vis, box_pts, colour_rgb, thickness=12)
        else:
            cv.polylines(vis, [box_pts], isClosed=True, color=colour_rgb, thickness=6)

        # Numbered circle at the bbox centre (fallback disambiguation)
        cx, cy = int(center[0]), int(center[1])
        cv.circle(vis, (cx, cy), 28, (0, 0, 0), -1)          # black halo
        cv.circle(vis, (cx, cy), 24, colour_rgb, -1)         # colour fill
        cv.putText(
            vis, str(i + 1), (cx - 10, cy + 8),
            cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv.LINE_AA,
        )

    return vis


def plot_bbox_analysis(
    bboxes: list[tuple],
    outlier_analysis: dict,
    tray_img: np.ndarray,
    image_label: str | None = None,
) -> None:
    """Produce one figure for the image with bar-chart property spread.

    Args:
        bboxes:           Full list of (center, size, angle, area) tuples from
                          the pipeline (pre-cut, same as passed to visualize).
        outlier_analysis: Dict from segment_instruments; provides scores/grades.
        tray_img:         RGB image used as the left panel backdrop.
        image_label:      Optional string for the figure title.
    """
    if not bboxes:
        print(f'[bbox_bar_analysis] {image_label or "image"}: no bboxes — skipping.')
        return

    records, outlier_indices = _build_per_bbox_records(bboxes, outlier_analysis)
    n_bboxes = len(bboxes)
    n_props  = len(_PROPERTIES)

    title = f'BBox property analysis — {n_bboxes} bbox(es)'
    if image_label:
        title = f'{image_label}  ·  {title}'

    fig = plt.figure(figsize=(22, max(6, 2.2 * n_props)))
    fig.suptitle(title, fontsize=12, fontweight='bold', y=0.98)

    outer_gs = gridspec.GridSpec(
        1, 3, figure=fig,
        width_ratios=[2.5, n_props, 1.8],
        wspace=0.06,
        top=0.93, bottom=0.07, left=0.02, right=0.98,
    )

    # ── Left: tray image with all bboxes ─────────────────────────────────────
    ax_img = fig.add_subplot(outer_gs[0])
    vis    = _draw_tray_with_bboxes(tray_img, bboxes, outlier_indices)
    ax_img.imshow(vis)
    ax_img.set_title(
        f'All bboxes  (contour colour = bar colour · '
        f'{len(outlier_indices)} outlier(s): thick dashed)',
        fontsize=9, fontweight='bold', pad=4,
    )
    ax_img.axis('off')

    # ── Centre: one horizontal-bar subplot per property ───────────────────────
    props_gs = gridspec.GridSpecFromSubplotSpec(
        n_props, 1,
        subplot_spec=outer_gs[1],
        hspace=0.55,
    )

    y_positions = np.arange(n_bboxes)
    bar_labels  = [str(i + 1) for i in range(n_bboxes)]

    for prop_row, (key, label, _higher_is_outlier) in enumerate(_PROPERTIES):
        ax_bar = fig.add_subplot(props_gs[prop_row])

        values  = np.array([rec[key] for rec in records], dtype=float)
        colours = [_colour(i) for i in range(n_bboxes)]

        # Gold border + band for outlier bboxes
        edge_colours = ['gold' if i in outlier_indices else 'none'
                        for i in range(n_bboxes)]
        edge_widths  = [2.0   if i in outlier_indices else 0.0
                        for i in range(n_bboxes)]

        bars = ax_bar.barh(
            y_positions, values,
            color=colours, edgecolor=edge_colours, linewidth=edge_widths,
            height=0.7,
        )

        x_max = np.nanmax(values) if np.any(np.isfinite(values)) else 1.0
        if x_max == 0:
            x_max = 1.0

        for bar, val in zip(bars, values):
            if np.isfinite(val):
                text_x = val + x_max * 0.02
                fmt    = f'{val:.0f}' if key == 'bbox_area' else f'{val:.2f}'
                ax_bar.text(
                    text_x, bar.get_y() + bar.get_height() / 2,
                    fmt,
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

        # Gold background band for each outlier row
        for oi in outlier_indices:
            ax_bar.axhspan(oi - 0.5, oi + 0.5,
                           facecolor='gold', alpha=0.10, zorder=0)

    # ── Right: compact numeric table ─────────────────────────────────────────
    ax_tbl = fig.add_subplot(outer_gs[2])
    ax_tbl.axis('off')

    col_labels = ['#', 'grade', 'area_sc', 'fill_sc', 'AR', 'fill%', 'area']
    tbl_data   = []
    for i, rec in enumerate(records):
        tbl_data.append([
            str(i + 1),
            f'{rec["grade"]:.3f}'        if np.isfinite(rec['grade'])        else '—',
            f'{rec["area_score"]:.3f}'   if np.isfinite(rec['area_score'])   else '—',
            f'{rec["fill_score"]:.3f}'   if np.isfinite(rec['fill_score'])   else '—',
            f'{rec["aspect_ratio"]:.2f}',
            f'{rec["fill_ratio"]:.1f}',
            f'{rec["bbox_area"]:.0f}',
        ])

    table = ax_tbl.table(
        cellText=tbl_data,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.3)

    for r in range(n_bboxes):
        # Colour the # column cell with the item's palette colour
        table[r + 1, 0].set_facecolor(to_rgba(_colour(r), 0.85))
        table[r + 1, 0].set_text_props(color='white', fontweight='bold')
        if r in outlier_indices:
            for c in range(1, len(col_labels)):
                table[r + 1, c].set_facecolor(to_rgba('gold', 0.25))
                table[r + 1, c].set_text_props(fontweight='bold')

    for c in range(len(col_labels)):
        table[0, c].set_facecolor('#333333')
        table[0, c].set_text_props(color='white', fontweight='bold')

    ax_tbl.set_title('Exact values', fontsize=8, pad=4)

    # Outlier scenario annotation
    scenario  = outlier_analysis.get('scenario', '?')
    grade_thr = outlier_analysis.get('grade_cutoff', float('nan'))
    fig.text(
        0.5, 0.01,
        (f'gold = outlier bbox  ·  scenario: {scenario}  ·  '
         f'grade cutoff: {grade_thr:.2f}'),
        ha='center', va='bottom', fontsize=8, style='italic', color='#555555',
    )

    plt.show()
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

def _process_one(result: tuple, label: str) -> None:
    """Unpack a pipeline result tuple and call plot_bbox_analysis."""
    # result[:6] — _process_image now also returns crops/pred_labels/confidences
    # (classification) after these six; take only the segmentation outputs.
    tray_no_bg, _seg_binary, _seg_cut, _final_bboxes, outlier_analysis, _concave_pts = result[:6]

    # Reconstruct the full bbox list in original pipeline order by merging
    # outliers and normals, then sorting by their position in the combined pool.
    # run_pipeline does not re-export raw bboxes, but outlier_analysis stores
    # every bbox object under 'outliers' or 'normal'.  We sort by Python id()
    # to get a stable (if arbitrary) ordering consistent within one run.
    all_entries = (outlier_analysis.get('outliers', [])
                   + outlier_analysis.get('normal',   []))
    all_entries_sorted = sorted(all_entries, key=lambda e: id(e['bbox']))
    bboxes = [e['bbox'] for e in all_entries_sorted]

    n_outliers = len(outlier_analysis.get('outliers', []))
    print(f'\n[bbox_bar_analysis] {label}: '
          f'{len(bboxes)} bbox(es), {n_outliers} outlier(s).')

    plot_bbox_analysis(bboxes, outlier_analysis, tray_no_bg, image_label=label)


if __name__ == '__main__':
    # None   → one random image from ./Trays
    # 'all'  → every image in ./Trays (figures appear one by one; close each to advance)
    # or a specific path, e.g. './Trays/IMG_3353.jpg'
    IMAGE_PATH: str | None = None
    IMAGE_PATH = "/home/antonio/Documents/TFG-project/debug-utils/IMG_outlier_study2.JPG" #'all'
    if IMAGE_PATH == 'all':
        tray_root = os.path.join(os.path.dirname(__file__), '..', 'Trays3')
        tray_paths = sorted(
            os.path.join(wd, f)
            for wd, _, files in os.walk(tray_root)
            for f in files
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
        )
        # Process one image at a time: run_pipeline(image_path='all') only returns
        # lightweight summary dicts (to bound memory across the whole dataset),
        # which can't be plotted. A per-image call returns the full result tuple
        # and keeps memory bounded to a single image.
        for path in tray_paths:
            res = run_pipeline(debugging=False, image_path=path)
            _process_one(res, os.path.basename(path))
    else:
        result = run_pipeline(debugging=False, image_path=IMAGE_PATH)
        label  = os.path.basename(IMAGE_PATH) if IMAGE_PATH else 'random image'
        _process_one(result, label)
