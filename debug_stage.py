# debug_stage.py
# Ad-hoc per-stage debugging: run the pipeline up to a chosen stage on one
# image and inspect the result, without running the full pipeline.
#
# Shows at most 3 panels: the original input, the preprocessed image, and the
# result for the chosen STAGE with its bounding boxes drawn on. (STAGEs before
# preprocessing show the input plus that single intermediate.)
#
# Usage:
#   1. Edit IMAGE and STAGE below.
#   2. python debug_stage.py
#
# STAGE options (each runs everything before it too):
#   "roi"        → get_ROI_from_color
#   "binary"     → + binarize_image
#   "tray"       → + get_tray_crop
#   "reflection" → + detect_specular_reflections
#   "preprocess" → + remove_blue_background      (full preprocessing)
#   "segment"    → + segment_instruments         (initial bboxes + outliers)
#   "concave"    → + detect_concave_points       (cut candidates on fused blobs)
#   "separate"   → + apply_concave_cuts + find_bboxes  (split fused tools)
#   "classify"   → + per-tool ToolClassifier prediction (full pipeline)

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import math

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from config import PreprocessConfig
from pipeline.preprocess import (
    get_ROI_from_color, binarize_image, get_tray_crop,
    detect_specular_reflections, remove_blue_background,
)
from pipeline.segmentation import segment_instruments, find_bboxes
from pipeline.concave_points import detect_concave_points
from pipeline.concave_cut import select_best_concave_points, apply_concave_cuts

# ── Edit these ────────────────────────────────────────────────────────────────
IMAGE = './Trays2/IMG_1261.jpg'   # path to a single image
STAGE = 'separate'               # see header for options
SAVE  = False                    # also write each panel to ./debug_results/
NCOLS = 3                        # number of columns in the visualisation grid
# ──────────────────────────────────────────────────────────────────────────────

_ORDER = ['roi', 'binary', 'tray', 'reflection', 'preprocess',
          'segment', 'concave', 'separate', 'classify']


def _to_rgb(img: np.ndarray) -> np.ndarray:
    """Return a 3-channel RGB copy so coloured overlays are visible."""
    if img.ndim == 2:
        return cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    return img.copy()


def _line_thickness(img: np.ndarray) -> int:
    """Bbox/marker thickness scaled to the image (2px is invisible at 4284px wide)."""
    return max(3, int(max(img.shape[:2]) / 400))


def _draw_bboxes(img: np.ndarray, bboxes: list, color=(255, 0, 0)) -> np.ndarray:
    """Draw oriented bboxes ((center, size, angle, ...) tuples) onto a copy."""
    out = _to_rgb(img)
    t = _line_thickness(out)
    for bbox in bboxes:
        center, size, angle = bbox[0], bbox[1], bbox[2]
        pts = cv.boxPoints(((center[0], center[1]), size, angle)).astype(np.int32)
        cv.polylines(out, [pts], True, color, t)
    return out


def _draw_points(img: np.ndarray, concave_pts: dict, color=(255, 0, 255)) -> np.ndarray:
    """Draw concave candidate points (dict[idx → list of {'x','y'}]) onto a copy."""
    out = _to_rgb(img)
    r = _line_thickness(out) * 2
    for pts in concave_pts.values():
        for p in pts:
            cv.circle(out, (int(p['x']), int(p['y'])), r, color, -1)
    return out


def _draw_labeled_bboxes(img: np.ndarray, bboxes: list, labels: list,
                         confs: list, color=(0, 200, 0)) -> np.ndarray:
    """Draw oriented bboxes plus a 'name conf%' label at each bbox centre."""
    out = _to_rgb(img)
    t = _line_thickness(out)
    scale = max(out.shape[:2]) / 1600.0
    for bbox, name, conf in zip(bboxes, labels, confs):
        center, size, angle = bbox[0], bbox[1], bbox[2]
        pts = cv.boxPoints(((center[0], center[1]), size, angle)).astype(np.int32)
        cv.polylines(out, [pts], True, color, t)
        cv.putText(out, f'{name} {conf:.0%}', (int(center[0]), int(center[1])),
                   cv.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 0), t, cv.LINE_AA)
    return out


def _show(panels: dict) -> None:
    """Display (and optionally save) each named intermediate in a grid."""
    if SAVE:
        os.makedirs('./debug_results', exist_ok=True)
        label = os.path.basename(IMAGE)
        for name, img in panels.items():
            cmap = 'gray' if img.ndim == 2 else None
            plt.imsave(f'./debug_results/{label}_{name}.png', img, cmap=cmap)

    n      = len(panels)
    ncols  = max(1, min(NCOLS, n))
    nrows  = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, img) in zip(axes, panels.items()):
        ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
        ax.set_title(name)
        ax.axis('off')
    for ax in axes[n:]:           # hide unused cells
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def main() -> None:
    if STAGE not in _ORDER:
        raise ValueError(f'STAGE must be one of {_ORDER}, got {STAGE!r}')
    target = _ORDER.index(STAGE)

    cfg = PreprocessConfig()
    img_bgr = cv.imread(IMAGE)
    if img_bgr is None:
        raise FileNotFoundError(f'Could not load image: {IMAGE}')
    img = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    img = cv.resize(img, cfg.image_dims, interpolation=cv.INTER_AREA)

    # Only ever three panels: the original input, the preprocessed image the
    # segmentation actually sees, and the result for the chosen STAGE with its
    # bounding boxes drawn on. Early STAGEs (before preprocessing) show the input
    # plus that single intermediate instead.
    panels = {'input': img}

    # ── Preprocessing (early STAGEs return just that intermediate) ────────────
    roi_crop, _roi_mask, _roi_bbox = get_ROI_from_color(img, cfg)
    if target == _ORDER.index('roi'):
        panels['roi'] = roi_crop
        return _show(panels)

    binary_mask = binarize_image(roi_crop, cfg)
    if target == _ORDER.index('binary'):
        panels['binary'] = binary_mask
        return _show(panels)

    tray_masked, _tray_mask, _ = get_tray_crop(roi_crop, binary_mask, cfg)
    if target == _ORDER.index('tray'):
        panels['tray'] = tray_masked
        return _show(panels)

    reflection_mask = detect_specular_reflections(tray_masked, cfg)
    if target == _ORDER.index('reflection'):
        panels['reflection'] = reflection_mask
        return _show(panels)

    tray_no_bg = remove_blue_background(tray_masked, cfg)
    panels['preprocess'] = tray_no_bg
    if target == _ORDER.index('preprocess'):
        return _show(panels)

    # ── Core segmentation ─────────────────────────────────────────────────────
    seg_binary, bboxes, outlier_analysis = segment_instruments(tray_no_bg, cfg)
    outlier_bboxes = [e['bbox'] for e in outlier_analysis['outliers']]
    print(f'[debug] scenario={outlier_analysis["scenario"]}  '
          f'bboxes={len(bboxes)}  outliers={len(outlier_bboxes)}')
    if target == _ORDER.index('segment'):
        panels['result'] = _draw_bboxes(seg_binary, bboxes)
        return _show(panels)

    # ── Concave-point detection ───────────────────────────────────────────────
    concave_pts = detect_concave_points(seg_binary, outlier_bboxes, cfg)
    if target == _ORDER.index('concave'):
        panels['result'] = _draw_points(
            _draw_bboxes(seg_binary, outlier_bboxes), concave_pts)
        return _show(panels)

    # ── Tool separation: concave cut + re-segmentation ────────────────────────
    if outlier_bboxes:
        best_points    = select_best_concave_points(concave_pts)
        seg_binary_cut = apply_concave_cuts(seg_binary, outlier_bboxes,
                                            best_points, cfg)
        final_bboxes   = find_bboxes(seg_binary_cut, cfg)
    else:
        seg_binary_cut = seg_binary
        final_bboxes   = bboxes
    print(f'[debug] final tools={len(final_bboxes)} '
          f'(was {len(bboxes)} before separation)')
    if target == _ORDER.index('separate'):
        panels['result'] = _draw_bboxes(seg_binary_cut, final_bboxes,
                                        color=(0, 200, 0))
        return _show(panels)

    # ── Classification (full pipeline) ────────────────────────────────────────
    # Imported lazily so the lighter stages above don't pay the model load cost.
    from run_pipeline import _extract_tool_crop
    from classifier.classify import build_classifier, classify_crop
    from classifier.classifier_config import ClassifierConfig
    from utils.tool_names import display_name

    print('[debug] building classifier (runs once)…')
    classifier  = build_classifier(ClassifierConfig())
    class_names = classifier.class_names
    labels, confs = [], []
    for i, bbox in enumerate(final_bboxes):
        crop = _extract_tool_crop(roi_crop, bbox)
        idx, conf = classify_crop(classifier, crop)
        name = display_name(class_names[idx] if class_names else str(idx))
        labels.append(name)
        confs.append(conf)
        print(f'[debug]   #{i + 1}  {name}  {conf:.0%}')
    panels['result'] = _draw_labeled_bboxes(tray_masked, final_bboxes, labels, confs)
    return _show(panels)


if __name__ == '__main__':
    main()
