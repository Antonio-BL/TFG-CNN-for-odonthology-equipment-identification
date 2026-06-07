# debug-utils/outlier_symmetry_analysis.py
# Vertical-symmetry study of the OUTLIER (fused-candidate) bounding boxes.
#
# Why:
#   A wide-open scissor or forceps looks geometrically identical to two fused
#   tools — there is no size/fill cue that tells them apart.  The one thing that
#   does: an open scissor/forceps is MIRROR-SYMMETRIC about its long axis, while
#   two genuinely fused (different) tools are not.  This script measures that.
#
# What it does, per outlier candidate:
#   1. Split the outlier's oriented bbox in two with a line PARALLEL to the long
#      side, through the middle (the "symmetry edge") -> two sub-bounding boxes.
#   2. For each sub-box compute an image signature (log Hu moments of the
#      contour — a shape fingerprint) plus geometric stats (area, fill ratio).
#   3. Compare the two halves (mirror IoU + Hu-moment shape distance + area
#      ratio) to decide whether the tool is symmetric -> likely a single tool.
#
# Layout per image (EXACT same style as bbox_bar_analysis.py):
#   Left   — Tray image: outliers drawn with their split line + two sub-boxes.
#             The enclosing box is GREEN (symmetric → single tool) or RED
#             (asymmetric → genuinely fused), dashed.  Each sub-box's contour
#             uses its palette colour, matching its bars and table row; a
#             numbered circle is the fallback.
#   Centre — One horizontal-bar subplot per signature property; one bar per
#             sub-box.  Sub-boxes of an ASYMMETRIC outlier carry a gold band.
#   Right  — Compact numeric table + per-outlier symmetry verdict.
#
# Usage:
#   cd /home/antonio/Documents/TFG-project
#   python debug-utils/outlier_symmetry_analysis.py
#
# Override IMAGE_PATH at the bottom:
#   None   → one random image    ·   'all' → every image   ·   or a specific path

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import to_rgba

from run_pipeline import main as run_pipeline

# ── Palette (same as bbox_bar_analysis) ───────────────────────────────────────
_ITEM_COLOURS = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
    '#dcbeff', '#9a6324',
]

# Per-sub-box signature properties shown as bars.  (key, label)
_PROPERTIES = [
    ('hu1',  'Hu 1  (log)'),
    ('hu2',  'Hu 2  (log)'),
    ('hu3',  'Hu 3  (log)'),
    ('hu4',  'Hu 4  (log)'),
    ('hu5',  'Hu 5  (log)'),
    ('hu6',  'Hu 6  (log)'),
    ('hu7',  'Hu 7  (log)'),
    ('area', 'Contour area  [px²]'),
    ('fill', 'Fill ratio  [%]'),
]

# Symmetry verdict thresholds (a tool is "symmetric" → likely a single open
# scissor/forceps → should NOT be split).
_IOU_SYMMETRIC   = 0.55     # mirror-IoU at/above this ⇒ symmetric
_SHAPE_SYMMETRIC = 0.30     # Hu shape distance at/below this ⇒ symmetric


def _colour(idx: int) -> str:
    return _ITEM_COLOURS[idx % len(_ITEM_COLOURS)]


def _rgb(idx: int) -> tuple[int, int, int]:
    """Palette colour as an (R, G, B) tuple (vis images are RGB, not BGR)."""
    h = _colour(idx)
    return int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)


# ──────────────────────────────────────────────────────────────────────────────
# Geometry: oriented-bbox axes, sub-box corners, deskew
# ──────────────────────────────────────────────────────────────────────────────

def _bbox_axes(bbox: tuple):
    """Return (centre, long_unit, short_unit, long_len, short_len) for an oriented bbox."""
    center, size, angle = bbox[0], bbox[1], bbox[2]
    w, h = float(size[0]), float(size[1])
    a = np.deg2rad(angle)
    u = np.array([np.cos(a),  np.sin(a)])   # along the width side
    v = np.array([-np.sin(a), np.cos(a)])   # along the height side
    c = np.array(center, dtype=float)
    if w >= h:
        return c, u, v, w, h
    return c, v, u, h, w


def _half_corners(c, long_v, short_v, long_len, short_len, side: int) -> np.ndarray:
    """4 corners (int) of one half-box: full long side, half short side, on `side`."""
    hc = c + side * (short_len / 4.0) * short_v       # centre of this half
    hl = (long_len / 2.0) * long_v
    hs = (short_len / 4.0) * short_v
    return np.int32([hc - hl + hs, hc + hl + hs, hc + hl - hs, hc - hl - hs])


def _deskew(binary: np.ndarray, c, long_v, short_v, long_len, short_len) -> np.ndarray:
    """Warp the oriented bbox region to an upright crop (long axis horizontal).

    Row 0 of the crop corresponds to the +short_v side (sub-box 'A').
    """
    W = max(1, int(round(long_len)))
    H = max(1, int(round(short_len)))
    hl = (long_len / 2.0) * long_v
    hs = (short_len / 2.0) * short_v
    src = np.float32([c - hl + hs, c + hl + hs, c + hl - hs, c - hl - hs])
    dst = np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])
    M = cv.getPerspectiveTransform(src, dst)
    return cv.warpPerspective(binary, M, (W, H))


# ──────────────────────────────────────────────────────────────────────────────
# Signatures + symmetry
# ──────────────────────────────────────────────────────────────────────────────

def _signature(mask: np.ndarray) -> dict | None:
    """Image signature (log Hu moments) + geometry of the largest blob in `mask`."""
    cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv.contourArea)
    area = float(cv.contourArea(cnt))
    hu = cv.HuMoments(cv.moments(cnt)).flatten()
    # Log transform keeps the seven moments on a comparable, signed scale.
    hu_log = [float(-np.sign(h) * np.log10(abs(h))) if h != 0 else 0.0 for h in hu]
    box_area = float(mask.shape[0] * mask.shape[1])
    return {
        'contour': cnt,
        'area':    area,
        'fill':    (area / box_area * 100.0) if box_area > 0 else 0.0,
        'hu':      hu_log,
    }


def _analyse_outlier(binary: np.ndarray, bbox: tuple) -> dict:
    """Split one outlier bbox, build per-half signatures and symmetry metrics."""
    c, long_v, short_v, long_len, short_len = _bbox_axes(bbox)
    crop = _deskew(binary, c, long_v, short_v, long_len, short_len)

    H = crop.shape[0]
    h2 = H // 2
    top    = crop[:h2]
    bottom = crop[h2:2 * h2]                 # equal height for the mirror compare
    sig_a  = _signature(top)                 # 'A' = +short_v side
    sig_b  = _signature(bottom)              # 'B' = -short_v side

    # Mirror IoU: flip the bottom half up and overlap it with the top half.
    if top.size and bottom.size:
        bot_flip = np.flipud(bottom)
        inter = int(np.logical_and(top > 0, bot_flip > 0).sum())
        union = int(np.logical_or(top > 0, bot_flip > 0).sum())
        iou = inter / union if union else 0.0
    else:
        iou = 0.0

    # Hu-moment shape distance + area ratio between the two halves.
    if sig_a and sig_b:
        shape_dist = float(cv.matchShapes(sig_a['contour'], sig_b['contour'],
                                          cv.CONTOURS_MATCH_I1, 0.0))
        a_area, b_area = sig_a['area'], sig_b['area']
        area_ratio = (min(a_area, b_area) / max(a_area, b_area)
                      if max(a_area, b_area) > 0 else 0.0)
    else:
        shape_dist = float('nan')
        area_ratio = 0.0

    symmetric = (iou >= _IOU_SYMMETRIC) and (
        not np.isfinite(shape_dist) or shape_dist <= _SHAPE_SYMMETRIC)

    return {
        'bbox': bbox, 'axes': (c, long_v, short_v, long_len, short_len),
        'sig_a': sig_a, 'sig_b': sig_b,
        'iou': iou, 'shape_dist': shape_dist, 'area_ratio': area_ratio,
        'symmetric': symmetric,
    }


def _record(sig: dict | None) -> dict:
    """Flatten a half signature into a bar/table record (NaNs when empty)."""
    nan = float('nan')
    if sig is None:
        rec = {k: nan for k, _ in _PROPERTIES}
        return rec
    rec = {f'hu{i + 1}': sig['hu'][i] for i in range(7)}
    rec['area'] = sig['area']
    rec['fill'] = sig['fill']
    return rec


# ──────────────────────────────────────────────────────────────────────────────
# Drawing
# ──────────────────────────────────────────────────────────────────────────────

def _draw_dashed_polygon(img, pts, colour_rgb, thickness, dash=34, gap=26) -> None:
    """Dashed closed polygon (OpenCV has no native dash style)."""
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
                    colour_rgb, thickness, cv.LINE_AA)
            d += dash + gap


def _draw_tray(tray_img: np.ndarray, analyses: list[dict]) -> np.ndarray:
    """Tray image with each outlier's split line + two sub-boxes drawn."""
    vis = tray_img.copy()

    for o, an in enumerate(analyses):
        c, long_v, short_v, long_len, short_len = an['axes']
        verdict_rgb = (60, 220, 60) if an['symmetric'] else (235, 40, 40)  # green / red

        # Enclosing oriented bbox: dashed, coloured by the symmetry verdict.
        box_pts = np.int32(cv.boxPoints((an['bbox'][0], an['bbox'][1], an['bbox'][2])))
        _draw_dashed_polygon(vis, box_pts, verdict_rgb, thickness=10)

        # Symmetry edge: the mid-line parallel to the long side.
        p0 = (c - (long_len / 2.0) * long_v).astype(int)
        p1 = (c + (long_len / 2.0) * long_v).astype(int)
        cv.line(vis, tuple(p0), tuple(p1), (255, 255, 255), 4, cv.LINE_AA)

        # Two sub-boxes, solid, each in its own palette colour (matches its bars).
        for k, side in enumerate((+1, -1)):
            item = 2 * o + k
            corners = _half_corners(c, long_v, short_v, long_len, short_len, side)
            cv.polylines(vis, [corners], True, _rgb(item), 6)

            hc = (c + side * (short_len / 4.0) * short_v).astype(int)
            cv.circle(vis, tuple(hc), 30, (0, 0, 0), -1)
            cv.circle(vis, tuple(hc), 26, _rgb(item), -1)
            cv.putText(vis, str(item + 1), (hc[0] - 12, hc[1] + 9),
                       cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv.LINE_AA)

    return vis


# ──────────────────────────────────────────────────────────────────────────────
# Figure
# ──────────────────────────────────────────────────────────────────────────────

def plot_symmetry_analysis(
    seg_binary: np.ndarray,
    outlier_analysis: dict,
    tray_img: np.ndarray,
    image_label: str | None = None,
) -> None:
    """One figure: outlier sub-boxes + per-half signatures + symmetry verdict."""
    outliers = [e['bbox'] for e in outlier_analysis.get('outliers', [])]
    if not outliers:
        print(f'[outlier_symmetry] {image_label or "image"}: no outlier candidates — skipping.')
        return

    analyses = [_analyse_outlier(seg_binary, bb) for bb in outliers]

    # One record per sub-box (2 per outlier), in item order A,B,A,B,...
    records: list[dict] = []
    asym_items: set[int] = set()
    for o, an in enumerate(analyses):
        records.append(_record(an['sig_a']))
        records.append(_record(an['sig_b']))
        if not an['symmetric']:
            asym_items.update({2 * o, 2 * o + 1})

    n_items = len(records)
    n_props = len(_PROPERTIES)

    title = f'Outlier vertical-symmetry analysis — {len(outliers)} outlier(s)'
    if image_label:
        title = f'{image_label}  ·  {title}'

    fig = plt.figure(figsize=(22, max(6, 2.2 * n_props)))
    fig.suptitle(title, fontsize=12, fontweight='bold', y=0.98)

    outer_gs = gridspec.GridSpec(
        1, 3, figure=fig,
        width_ratios=[2.5, n_props, 1.8],
        wspace=0.06, top=0.93, bottom=0.07, left=0.02, right=0.98,
    )

    # ── Left: tray image with sub-boxes ──────────────────────────────────────
    ax_img = fig.add_subplot(outer_gs[0])
    ax_img.imshow(_draw_tray(tray_img, analyses))
    ax_img.set_title(
        'Outliers split on the symmetry edge  ·  green = symmetric (1 tool) · '
        'red = asymmetric (fused)',
        fontsize=9, fontweight='bold', pad=4,
    )
    ax_img.axis('off')

    # ── Centre: one bar subplot per signature property ───────────────────────
    props_gs = gridspec.GridSpecFromSubplotSpec(n_props, 1, subplot_spec=outer_gs[1], hspace=0.55)
    y_pos      = np.arange(n_items)
    bar_labels = [str(i + 1) for i in range(n_items)]
    colours    = [_colour(i) for i in range(n_items)]
    edge_cols  = ['gold' if i in asym_items else 'none' for i in range(n_items)]
    edge_wids  = [2.0    if i in asym_items else 0.0    for i in range(n_items)]

    for prop_row, (key, label) in enumerate(_PROPERTIES):
        ax = fig.add_subplot(props_gs[prop_row])
        values = np.array([rec[key] for rec in records], dtype=float)

        ax.barh(y_pos, values, color=colours, edgecolor=edge_cols,
                linewidth=edge_wids, height=0.7)

        finite = values[np.isfinite(values)]
        vmin = min(0.0, float(finite.min())) if finite.size else 0.0
        vmax = max(0.0, float(finite.max())) if finite.size else 1.0
        span = (vmax - vmin) or 1.0
        ax.axvline(0, color='#bbbbbb', linewidth=0.6, zorder=0)
        for yi, val in zip(y_pos, values):
            if np.isfinite(val):
                fmt = f'{val:.0f}' if key == 'area' else f'{val:.2f}'
                ax.text(val + span * 0.02 * (1 if val >= 0 else -1), yi, fmt,
                        va='center', ha='left' if val >= 0 else 'right',
                        fontsize=6.5, color='#333333')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(bar_labels, fontsize=7)
        ax.set_xlabel(label, fontsize=7, labelpad=2)
        ax.tick_params(axis='x', labelsize=6)
        ax.set_xlim(vmin - span * 0.15, vmax + span * 0.35)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for i in asym_items:
            ax.axhspan(i - 0.5, i + 0.5, facecolor='gold', alpha=0.10, zorder=0)

    # ── Right: numeric table + per-outlier verdict ───────────────────────────
    ax_tbl = fig.add_subplot(outer_gs[2])
    ax_tbl.axis('off')

    col_labels = ['#', 'out', 'side', 'hu1', 'hu2', 'area', 'fill%']
    tbl_data   = []
    for o, an in enumerate(analyses):
        for k, (side, rec) in enumerate((('A', records[2 * o]), ('B', records[2 * o + 1]))):
            tbl_data.append([
                str(2 * o + k + 1), str(o + 1), side,
                f'{rec["hu1"]:.2f}' if np.isfinite(rec['hu1']) else '—',
                f'{rec["hu2"]:.2f}' if np.isfinite(rec['hu2']) else '—',
                f'{rec["area"]:.0f}' if np.isfinite(rec['area']) else '—',
                f'{rec["fill"]:.1f}' if np.isfinite(rec['fill']) else '—',
            ])

    table = ax_tbl.table(cellText=tbl_data, colLabels=col_labels,
                         cellLoc='center', loc='upper center')
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.3)
    for i in range(n_items):
        table[i + 1, 0].set_facecolor(to_rgba(_colour(i), 0.85))
        table[i + 1, 0].set_text_props(color='white', fontweight='bold')
        if i in asym_items:
            for col in range(1, len(col_labels)):
                table[i + 1, col].set_facecolor(to_rgba('gold', 0.25))
    for col in range(len(col_labels)):
        table[0, col].set_facecolor('#333333')
        table[0, col].set_text_props(color='white', fontweight='bold')
    ax_tbl.set_title('Per-sub-box signature', fontsize=8, pad=4)

    # Per-outlier verdict block under the table.
    lines = []
    for o, an in enumerate(analyses):
        tag = 'SYMMETRIC → 1 tool' if an['symmetric'] else 'ASYMMETRIC → fused'
        sd  = f'{an["shape_dist"]:.3f}' if np.isfinite(an['shape_dist']) else '—'
        lines.append(f'Outlier {o + 1}: {tag}   '
                     f'(mirror IoU {an["iou"]:.2f} · shape dist {sd} · '
                     f'area ratio {an["area_ratio"]:.2f})')
    ax_tbl.text(0.0, -0.02, '\n'.join(lines), transform=ax_tbl.transAxes,
                ha='left', va='top', fontsize=8, family='monospace')

    fig.text(
        0.5, 0.01,
        (f'gold = sub-boxes of an asymmetric outlier  ·  '
         f'symmetric if mirror-IoU ≥ {_IOU_SYMMETRIC} and shape dist ≤ {_SHAPE_SYMMETRIC}'),
        ha='center', va='bottom', fontsize=8, style='italic', color='#555555',
    )

    plt.show()
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

def _process_one(result: tuple, label: str) -> None:
    """Unpack a pipeline result tuple and run the symmetry analysis."""
    tray_no_bg, seg_binary, _seg_cut, _final_bboxes, outlier_analysis, _concave_pts = result[:6]
    n_out = len(outlier_analysis.get('outliers', []))
    print(f'\n[outlier_symmetry] {label}: {n_out} outlier candidate(s).')
    plot_symmetry_analysis(seg_binary, outlier_analysis, tray_no_bg, image_label=label)


if __name__ == '__main__':
    # None → one random image · 'all' → every image (close each figure to advance)
    # or a specific path, e.g. './Trays3/IMG_1303.jpg'
    IMAGE_PATH: str | None = None
    IMAGE_PATH = 'all'
    if IMAGE_PATH == 'all':
        tray_root = os.path.join(os.path.dirname(__file__), '..', 'Trays3')
        tray_paths = sorted(
            os.path.join(wd, f)
            for wd, _, files in os.walk(tray_root)
            for f in files
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
        )
        # One image at a time: full result tuple, memory bounded to a single image.
        for path in tray_paths:
            res = run_pipeline(debugging=False, image_path=path)
            _process_one(res, os.path.basename(path))
    else:
        result = run_pipeline(debugging=False, image_path=IMAGE_PATH)
        label  = os.path.basename(IMAGE_PATH) if IMAGE_PATH else 'random image'
        _process_one(result, label)
