# visualize.py
# Debug visualisation for the TFG dental-instrument segmentation pipeline.
#
# Public functions:
#   plot_segmentation_results — 6-panel overview of one image's pipeline run.
#
# Architecture: consumer-only.  This module receives pre-computed data and
# draws it; it never produces data nor returns results to the pipeline.
# Algorithm modules never import from here.

import math
import pathlib

import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use('Qt5Agg')   # interactive backend; requires PyQt5
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines   as mlines
import matplotlib.gridspec as gridspec

from config       import PreprocessConfig
from utils        import edge_Laplace
from concave_cut  import _concave_grade


# Threshold applied to |Laplacian(seg_binary)| when drawing the edge overlay
# in Panel [1,0].  1.0 is the standard value.
_EDGE_DISPLAY_THRESHOLD: float = 1.0


def plot_segmentation_results(
    tray_masked: np.ndarray,
    tray_no_bg: np.ndarray,
    seg_binary: np.ndarray,
    bboxes: list[tuple],
    reflection_mask: np.ndarray | None = None,
    image_label: str | None = None,
    outlier_analysis: dict | None = None,
    img_rgb: np.ndarray | None = None,
    roi_bbox: tuple | None = None,
    cfg: PreprocessConfig | None = None,
    concave_points: dict | None = None,
    seg_binary_cut: np.ndarray | None = None,
    final_bboxes: list[tuple] | None = None,
) -> None:
    """Show a 2×3 overview figure for one image's segmentation result.

    Panel layout:
      [0,0] Original image + ROI bounding box (or tray_masked if no img_rgb).
      [0,1] Specular reflection mask (or background-removed image if missing).
      [0,2] Sauvola binary mask (pre-cut).
      [1,0] Laplacian edges + initial oriented bboxes
            (red = outlier/fused, green = normal, cyan dot = concave point).
      [1,1] When outliers exist AND seg_binary_cut + final_bboxes are provided:
            Post-cut binary mask with the final per-instrument bboxes drawn in
            lime, and the chosen cut point (max chord) marked as a cyan star.
            Otherwise: edges + bboxes (same as [1,0]).
      [1,2] Per-bbox stats table.

    Args:
        tray_masked:      RGB tray image after ROI crop (H, W, 3).
        tray_no_bg:       RGB image after blue-background removal (H, W, 3).
        seg_binary:       uint8 binary mask (H, W); instruments = 255 (pre-cut).
        bboxes:           List of initial (center, size, angle, area) 4-tuples.
                          Outliers among these are highlighted in red.
        reflection_mask:  Optional uint8 mask of detected specular reflections.
        image_label:      Optional string shown as the figure title.
        outlier_analysis: Dict from _analyze_bbox_outliers; used to colour
                          bboxes and populate the stats table.
        img_rgb:          Optional full-resolution RGB image for Panel [0,0].
        roi_bbox:         Optional (x, y, w, h) green rectangle on Panel [0,0].
        cfg:              Optional PreprocessConfig (unused; kept for future).
        concave_points:   Optional dict[bbox_idx → list of point dicts] from
                          detect_concave_points.  Each dict has 'x', 'y',
                          'kappa', 'chord_len', etc.  All candidates are drawn
                          as cyan dots on Panel [1,0]; the candidate with the
                          maximum chord_len per outlier is marked as a cyan
                          star on Panel [1,1] (the actual cut point).
        seg_binary_cut:   Optional uint8 binary mask AFTER apply_concave_cuts.
                          Used as the background of Panel [1,1] when outliers
                          exist.
        final_bboxes:     Optional list of post-cut bboxes from
                          find_bboxes(seg_binary_cut, cfg).  Drawn on Panel
                          [1,1] as lime polygons — one per separated tool.
    """
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    if image_label:
        fig.suptitle(image_label, fontsize=13, fontweight='bold')

    # Set of bbox object ids classified as outliers (for red highlight)
    outlier_ids: set[int] = set()
    if outlier_analysis and outlier_analysis.get('outliers'):
        outlier_ids = {id(e['bbox']) for e in outlier_analysis['outliers']}

    # ── Panel [0,0]: original image + ROI rectangle ─────────────────────────
    if img_rgb is not None and roi_bbox is not None:
        roi_viz = img_rgb.copy()
        x0, y0, rw, rh = roi_bbox
        cv.rectangle(roi_viz, (x0, y0), (x0 + rw, y0 + rh), (0, 255, 0), thickness=4)
        axs.flat[0].imshow(roi_viz)
        axs.flat[0].set_title('Original image + ROI')
    else:
        axs.flat[0].imshow(tray_masked)
        axs.flat[0].set_title('Tray masked')
    axs.flat[0].axis('off')

    # ── Panel [0,1]: specular reflections ────────────────────────────────────
    if reflection_mask is not None:
        axs.flat[1].imshow(reflection_mask, cmap='gray')
        axs.flat[1].set_title('Specular reflections removed')
    else:
        axs.flat[1].imshow(tray_no_bg)
        axs.flat[1].set_title('Background removed')
    axs.flat[1].axis('off')

    # ── Panel [0,2]: Sauvola binary mask (pre-cut) ───────────────────────────
    axs.flat[2].imshow(seg_binary, cmap='gray')
    axs.flat[2].set_title('Sauvola binary mask (pre-cut)')
    axs.flat[2].axis('off')

    # ── Panel [1,0]: Laplacian edges + initial bboxes ───────────────────────
    edge_img    = edge_Laplace(seg_binary.astype(np.float32))
    edge_binary = np.abs(edge_img) > _EDGE_DISPLAY_THRESHOLD
    overlay     = cv.cvtColor(seg_binary, cv.COLOR_GRAY2RGB)
    h_img, w_img = seg_binary.shape

    for bbox in bboxes:
        center, size, angle = bbox[0], bbox[1], bbox[2]
        bbox_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cv.fillPoly(
            bbox_mask, [np.int32(cv.boxPoints((center, size, angle)))], 255,
        )
        colour_rgb = (255, 0, 0) if id(bbox) in outlier_ids else (0, 255, 0)
        overlay[edge_binary & (bbox_mask > 0)] = colour_rgb

    bbox_ax = axs.flat[3]
    bbox_ax.imshow(overlay)
    bbox_ax.set_title(
        f'Initial bboxes  ({len(bboxes)} detected, '
        f'{len(outlier_ids)} fused)'
    )
    bbox_ax.axis('off')

    for i, bbox in enumerate(bboxes):
        center, size, angle, area = bbox
        is_outlier = id(bbox) in outlier_ids
        colour = 'red' if is_outlier else 'lime'
        box_pts = np.int32(cv.boxPoints(((center[0], center[1]), size, angle)))
        bbox_ax.add_patch(mpatches.Polygon(
            box_pts, closed=True,
            linewidth=2, edgecolor=colour, facecolor='none',
        ))
        w, h = size
        lbl = f'#{i + 1}  {w:.0f}×{h:.0f}  {angle:.1f}°  ({area:,} px)'
        if is_outlier:
            lbl += '  ⚠ outlier'
        bbox_ax.text(
            center[0], center[1] - 20, lbl,
            color=colour, fontsize=7, fontweight='bold',
            bbox=dict(facecolor='black', alpha=0.45, pad=1, edgecolor='none'),
        )

    # Concave-point markers on Panel [1,0] (every candidate)
    if concave_points is not None:
        for _bidx, points in concave_points.items():
            for pt in points:
                px, py = pt['x'], pt['y']
                bbox_ax.plot(px, py, 'o',
                             color='cyan', markersize=6, markeredgecolor='black',
                             markeredgewidth=0.6, zorder=10)
                bbox_ax.text(
                    px + 8, py,
                    f'κ={pt["kappa"]:.2f}  chord={pt["chord_len"]:.1f}',
                    color='cyan', fontsize=6, va='center', zorder=11,
                    bbox=dict(facecolor='black', alpha=0.45, pad=1, edgecolor='none'),
                )

    # Legend on Panel [1,0]
    _legend = [
        mpatches.Patch(edgecolor='lime', facecolor='none', linewidth=2,
                       label='Normal'),
        mpatches.Patch(edgecolor='red',  facecolor='none', linewidth=2,
                       label='Outlier / fused'),
    ]
    if concave_points is not None:
        _legend.append(
            mlines.Line2D([], [], marker='o', color='cyan', markersize=6,
                          markeredgecolor='black', markeredgewidth=0.6,
                          linewidth=0, label='Concave point'),
        )
    bbox_ax.legend(handles=_legend, loc='upper right', fontsize=7,
                   facecolor='#1a1a1a', labelcolor='white', framealpha=0.85)

    # ── Panel [1,1]: post-cut bboxes (or fallback when no anomaly) ──────────
    cut_ax       = axs.flat[4]
    has_outliers = bool(outlier_ids)
    show_cut     = (has_outliers
                    and seg_binary_cut is not None
                    and final_bboxes   is not None)

    if show_cut:
        cut_ax.imshow(seg_binary_cut, cmap='gray')
        cut_ax.set_title(
            f'After concave cut — {len(final_bboxes)} final instrument(s)'
        )

        for i, bbox in enumerate(final_bboxes):
            center, size, angle, area = bbox
            box_pts = np.int32(cv.boxPoints(((center[0], center[1]), size, angle)))
            cut_ax.add_patch(mpatches.Polygon(
                box_pts, closed=True,
                linewidth=2, edgecolor='lime', facecolor='none',
            ))
            w, h = size
            cut_ax.text(
                center[0], center[1] - 20,
                f'#{i + 1}  {w:.0f}×{h:.0f}  ({area:,} px)',
                color='lime', fontsize=7, fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.45, pad=1, edgecolor='none'),
            )

        # Mark the best concave point per outlier (the cut point) as a star
        if concave_points is not None:
            for _bidx, points in concave_points.items():
                if not points:
                    continue
                best = max(points, key=lambda p: _concave_grade(p, points))
                cut_ax.plot(best['x'], best['y'], '*',
                            color='cyan', markersize=14, markeredgecolor='black',
                            markeredgewidth=0.9, zorder=10)

        _cut_legend = [
            mpatches.Patch(edgecolor='lime', facecolor='none', linewidth=2,
                           label='Final bbox'),
        ]
        if concave_points is not None:
            _cut_legend.append(
                mlines.Line2D([], [], marker='*', color='cyan', markersize=14,
                              markeredgecolor='black', markeredgewidth=0.9,
                              linewidth=0, label='Cut point (best grade)'),
            )
        cut_ax.legend(handles=_cut_legend, loc='upper right', fontsize=7,
                      facecolor='#1a1a1a', labelcolor='white', framealpha=0.85)
    else:
        # No outliers — repeat the edges + bboxes view from [1,0]
        cut_ax.imshow(overlay)
        cut_ax.set_title(f'No outliers — {len(bboxes)} instrument(s)')
        for bbox in bboxes:
            center, size, angle, _ = bbox
            colour = 'red' if id(bbox) in outlier_ids else 'lime'
            box_pts = np.int32(cv.boxPoints(((center[0], center[1]), size, angle)))
            cut_ax.add_patch(mpatches.Polygon(
                box_pts, closed=True,
                linewidth=2, edgecolor=colour, facecolor='none',
            ))
    cut_ax.axis('off')

    # ── Panel [1,2]: per-bbox stats table ────────────────────────────────────
    text_ax = axs.flat[5]
    text_ax.set_xticks([])
    text_ax.set_yticks([])
    for spine in text_ax.spines.values():
        spine.set_visible(False)
    text_ax.set_facecolor('#f0f0f0')
    text_ax.set_title('BBox stats', fontsize=9)

    # Build id → (entry, label) lookup from outlier_analysis
    bbox_info: dict[int, tuple] = {}
    if outlier_analysis:
        for entry in outlier_analysis.get('outliers', []):
            bbox_info[id(entry['bbox'])] = (entry, 'outlier')
        for entry in outlier_analysis.get('normal', []):
            bbox_info[id(entry['bbox'])] = (entry, 'normal')

    nan = float('nan')
    header = f"  {'#':>6}  {'area_sc':>7}  {'fill_sc':>7}  {'grade':>6}  label"
    sep    = '  ' + '─' * (len(header) - 2)
    lines  = [header, sep]

    if outlier_analysis:
        med_a = outlier_analysis.get('median_area_score', nan)
        med_f = outlier_analysis.get('median_fill_score', nan)
        med_g = outlier_analysis.get('median_grade',      nan)
        thr_g = outlier_analysis.get('grade_cutoff',       nan)
        lines.append(
            f"  {'median':>6}  {med_a:>7.2f}  {med_f:>7.2f}  {med_g:>6.2f}  —"
        )
        lines.append(
            f"  {'thresh':>6}  {'':>7}  {'':>7}  {'>' + f'{thr_g:.2f}':>6}  —"
        )
        lines.append(sep)

    for i, bbox in enumerate(bboxes):
        bid = id(bbox)
        if bid in bbox_info:
            entry, label = bbox_info[bid]
            a_sc  = entry.get('area_score', nan)
            f_sc  = entry.get('fill_score', nan)
            grade = entry.get('grade',      nan)
        else:
            a_sc = f_sc = grade = nan
            label = '?'
        lines.append(
            f"  #{i + 1:<5}  {a_sc:>7.2f}  {f_sc:>7.2f}  {grade:>6.2f}  {label}"
        )

    # Secondary filter summary
    lines.append('')
    if outlier_analysis:
        prim      = outlier_analysis.get('primary_candidates', [])
        sec_cut   = outlier_analysis.get('secondary_cutoff',    0.0)
        max_cg    = outlier_analysis.get('max_candidate_grade', 0.0)
        sec_ratio = outlier_analysis.get('secondary_ratio',     nan)
        outlier_ids_local = {id(e['bbox'])
                             for e in outlier_analysis.get('outliers', [])}
        bbox_id_to_idx = {id(b): i for i, b in enumerate(bboxes)}

        triggered = len(prim) >= 2
        trigger_lbl = (
            'TRIGGERED'
            if triggered
            else f'not triggered ({len(prim)} primary candidate(s))'
        )
        lines.append(f'  Secondary filter — {trigger_lbl}')

        if triggered:
            lines.append(
                f'  max_grade: {max_cg:.2f}   cutoff: {sec_cut:.2f}'
                f'   ratio: {sec_ratio:.1f}'
            )
            sec_header = f"  {'#':>6}  {'grade':>6}  status"
            sec_sep    = '  ' + '─' * (len(sec_header) - 2)
            lines += [sec_header, sec_sep]
            for cand in sorted(prim, key=lambda e: e['grade'], reverse=True):
                bid = id(cand['bbox'])
                idx = bbox_id_to_idx.get(bid, -1)
                lbl = 'OUTLIER' if bid in outlier_ids_local else 'demoted'
                lines.append(f'  #{idx + 1:<5}  {cand["grade"]:>6.2f}  {lbl}')

    # Final-count summary appended at the bottom
    if show_cut:
        lines.append('')
        lines.append(f'  Initial → Final : {len(bboxes)} → {len(final_bboxes)}')

    text_ax.text(
        0.05, 0.95, '\n'.join(lines),
        transform=text_ax.transAxes,
        ha='left', va='top',
        fontsize=8, family='monospace',
        color='black',
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    plt.close()


def plot_pipeline_result(
    tray_rgb: np.ndarray,
    final_bboxes: list[tuple],
    crops: list[np.ndarray],
    pred_labels: list[str],
    confidences: list[float],
    image_label: str | None = None,
    save_path: str | None = None,
    warn_threshold: float = 0.6,
) -> None:
    """2-panel figure: annotated tray image (left) + deskewed crop gallery (right).

    Left panel: tray_rgb with each final bbox drawn as a polygon labelled
    "{name}  {conf:.0%}" at the box centre.  Lime = conf >= warn_threshold,
    orange = below threshold.

    Right panel: grid gallery of deskewed crops, each subplot titled with the
    predicted name and confidence (orange title when conf < warn_threshold).

    Saves to save_path when given (parent dirs created automatically),
    then always calls plt.show().
    """
    n = len(crops)
    ncols = min(5, max(1, n))
    nrows = max(1, math.ceil(n / ncols)) if n > 0 else 1

    fig = plt.figure(figsize=(14, max(6, 3 * nrows)))
    if image_label:
        fig.suptitle(image_label, fontsize=13, fontweight='bold')

    outer = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.2, 1], wspace=0.05)

    # ── Left panel: annotated tray ───────────────────────────────────────────
    ax_tray = fig.add_subplot(outer[0])
    ax_tray.imshow(tray_rgb)
    ax_tray.set_title('Segmentation + classification', fontsize=10)
    ax_tray.axis('off')

    for bbox, name, conf in zip(final_bboxes, pred_labels, confidences):
        center, size, angle, _ = bbox
        box_pts = np.int32(cv.boxPoints(((center[0], center[1]), size, angle)))
        colour = 'lime' if conf >= warn_threshold else 'orange'
        ax_tray.add_patch(mpatches.Polygon(
            box_pts, closed=True,
            linewidth=2, edgecolor=colour, facecolor='none',
        ))
        ax_tray.text(
            center[0], center[1],
            f'{name}  {conf:.0%}',
            color=colour, fontsize=8, fontweight='bold',
            ha='center', va='center',
            bbox=dict(facecolor='black', alpha=0.55, pad=2, edgecolor='none'),
        )

    # ── Right panel: crop gallery ────────────────────────────────────────────
    if n > 0:
        inner = gridspec.GridSpecFromSubplotSpec(
            nrows, ncols, subplot_spec=outer[1], hspace=0.5, wspace=0.3,
        )
        for i, (crop, name, conf) in enumerate(zip(crops, pred_labels, confidences)):
            ax = fig.add_subplot(inner[i])
            ax.imshow(crop)
            title_colour = 'black' if conf >= warn_threshold else 'darkorange'
            ax.set_title(f'{name}\n{conf:.0%}', fontsize=8, color=title_colour)
            ax.axis('off')
        for j in range(n, nrows * ncols):
            fig.add_subplot(inner[j]).axis('off')
    else:
        ax_empty = fig.add_subplot(outer[1])
        ax_empty.text(0.5, 0.5, 'No instruments detected',
                      ha='center', va='center', transform=ax_empty.transAxes,
                      fontsize=11, color='gray')
        ax_empty.axis('off')

    if save_path is not None:
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()
    plt.close()
