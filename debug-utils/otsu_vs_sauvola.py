# debug-utils/otsu_vs_sauvola.py
# Rough demo: how the segmentation behaves with Sauvola (the pipeline default)
# vs a plain global Otsu threshold.
#
# It runs the REAL preprocessing on each image, then binarises the
# background-removed tray two ways and compares the resulting masks + the
# oriented bounding boxes detected from each. One figure row per image:
#   [ preprocessed input ] [ Sauvola mask + boxes ] [ Otsu mask + boxes ]
#
# Usage:
#   python debug-utils/otsu_vs_sauvola.py
# Edit IMAGES / SHOW / SAVE below.

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import pathlib as _pathlib
import sys as _sys
_ROOT = _pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from config import PreprocessConfig
from pipeline.preprocess import (
    get_ROI_from_color, binarize_image, get_tray_crop, remove_blue_background,
)
from pipeline.segmentation import (
    _connect_edges, _binarize_tray, _find_bboxes, _filter_by_median_area,
)

# ── Edit these ────────────────────────────────────────────────────────────────
IMAGES = ['./Trays2/IMG_1261.jpg', './Trays2/IMG_1271.jpg', './Trays/IMG_3344.jpg']
SHOW   = True                                   # pop a window (ignored if headless)
SAVE   = './debug-utils/otsu_vs_sauvola.png'    # set to None to skip saving
# ──────────────────────────────────────────────────────────────────────────────


def _binarize_otsu(tray_no_bg, cfg):
    """Otsu drop-in for _binarize_tray: same morphology, global threshold instead.

    Sauvola adapts the threshold per pixel from the local mean/std; Otsu picks a
    single global threshold from the grey histogram. Everything after the
    threshold (dilate + open) is identical so the comparison is apples-to-apples.
    """
    gray = cv.cvtColor(tray_no_bg, cv.COLOR_RGB2GRAY)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    dilate_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, cfg.seg_close_kernel_dims)
    binary = cv.morphologyEx(binary, cv.MORPH_DILATE, dilate_kernel)
    open_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, cfg.bin_open_kernel_dims)
    binary = cv.morphologyEx(binary, cv.MORPH_OPEN, open_kernel)
    return binary


def _bboxes(binary, cfg):
    """Run the same bbox-detection + median-area filter the pipeline uses."""
    bb = _find_bboxes(binary, cfg.seg_min_contour_area)
    return _filter_by_median_area(bb, binary, cfg.seg_median_area_threshold)


def _draw(binary, bboxes, color=(0, 230, 0)):
    """Binary mask → RGB with oriented bboxes drawn (thickness scaled to size)."""
    out = cv.cvtColor(binary, cv.COLOR_GRAY2RGB)
    t = max(3, int(max(out.shape[:2]) / 400))
    for bbox in bboxes:
        pts = cv.boxPoints((bbox[0], bbox[1], bbox[2])).astype(np.int32)
        cv.polylines(out, [pts], True, color, t)
    return out


def _preprocess(path, cfg):
    """Real pipeline preprocessing up to the background-removed, edge-closed tray."""
    img = cv.cvtColor(cv.imread(path), cv.COLOR_BGR2RGB)
    img = cv.resize(img, cfg.image_dims, interpolation=cv.INTER_AREA)
    roi, _, _ = get_ROI_from_color(img, cfg)
    binary_mask = binarize_image(roi, cfg)
    tray, _, _ = get_tray_crop(roi, binary_mask, cfg)
    nbg = remove_blue_background(tray, cfg)
    return _connect_edges(nbg, cfg)


def main():
    cfg = PreprocessConfig()
    rows = []
    for path in IMAGES:
        if cv.imread(path) is None:
            print(f'[skip] cannot read {path}')
            continue
        tray = _preprocess(path, cfg)

        sa = _binarize_tray(tray, cfg)          # Sauvola (pipeline default)
        ot = _binarize_otsu(tray, cfg)          # Otsu (global)
        bb_sa, bb_ot = _bboxes(sa, cfg), _bboxes(ot, cfg)

        name = os.path.basename(path)
        print(f'{name:22s}  Sauvola={len(bb_sa):>2d} tools   Otsu={len(bb_ot):>2d} tools')
        rows.append((name, tray, _draw(sa, bb_sa), len(bb_sa),
                            _draw(ot, bb_ot), len(bb_ot)))

    if not rows:
        print('Nothing to show.')
        return

    fig, axes = plt.subplots(len(rows), 3, figsize=(13, 4.3 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, (name, tray, sa_img, n_sa, ot_img, n_ot) in enumerate(rows):
        for ax, im, title in (
            (axes[r, 0], tray,   f'{name}\ninput (preprocessed)'),
            (axes[r, 1], sa_img, f'Sauvola — {n_sa} tools'),
            (axes[r, 2], ot_img, f'Otsu — {n_ot} tools'),
        ):
            ax.imshow(im)
            ax.set_title(title, fontsize=10)
            ax.axis('off')
    plt.tight_layout()
    if SAVE:
        fig.savefig(SAVE, dpi=110, bbox_inches='tight')
        print(f'saved {SAVE}')
    if SHOW:
        plt.show()


if __name__ == '__main__':
    main()
