"""Generate pipeline-style crops from raw Tools/ images.

For each Tools/HerramientaN/*.jpg, run the *applicable* parts of the segmentation
pipeline (the tray-detection step is skipped because single-tool images have no
distinct tray contour) and save the perspective-corrected crop of the largest
detected bbox into pipeline_crops/HerramientaN/. The resulting training set
matches the distribution the classifier sees at inference time, where crops are
produced by _extract_tool_crop on segmented tray bboxes.

This module does NOT modify any pipeline file; it only re-uses their public
functions (get_ROI_from_color, remove_blue_background, segment_instruments,
_extract_tool_crop) in an order suited to single-tool input.
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pathlib
import sys
import time

import cv2 as cv
import numpy as np

from config import PreprocessConfig
from preprocess import get_ROI_from_color, remove_blue_background
from segmentation import segment_instruments
from run_pipeline import _extract_tool_crop
from progress import img_bar


SRC_ROOT = pathlib.Path("./Tools")
DST_ROOT = pathlib.Path("./pipeline_crops")


def _best_tool_bbox(bboxes: list) -> tuple | None:
    """Pick the bbox most likely to be a tool: elongated and large.

    `segment_instruments` can return blob-shaped artifacts (whole-cloth regions,
    shadows) alongside the tool itself; for single-tool training images the
    largest-by-area heuristic frequently selects those artifacts. Tools are
    characterised by a long-axis-dominant aspect ratio, so we filter to bboxes
    with long/short ≥ 3 and pick the one with the greatest long-side length.
    """
    candidates = []
    for b in bboxes:
        (_, _), (w, h), _, _ = b
        long_side, short_side = max(w, h), max(min(w, h), 1.0)
        if long_side / short_side >= 3.0:
            candidates.append((long_side, b))
    if candidates:
        return max(candidates, key=lambda t: t[0])[1]
    return max(bboxes, key=lambda b: b[3]) if bboxes else None


def _is_blue_background(img_bgr: np.ndarray, threshold: float = 0.3) -> bool:
    """True if a significant share of pixels match the surgical blue cloth."""
    hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
    h, s = hsv[..., 0], hsv[..., 1]
    blue_frac = ((h > 80) & (h < 140) & (s > 50)).mean()
    return blue_frac > threshold


def process_training_image(image_path: str, cfg: PreprocessConfig) -> np.ndarray | None:
    """Return a deskewed crop of the tool in image_path, or None on failure.

    Only blue-background training images are processed — the inference pipeline
    is built around blue-cloth segmentation, so a crop from a non-blue image
    would have a different appearance (white vs black background, different
    edge contrast) and contaminate the training distribution. All ten classes
    have at least seven blue-background images, so dropping the others still
    leaves every class represented.
    """
    img_bgr = cv.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(image_path)

    if not _is_blue_background(img_bgr):
        return None

    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    img_rgb = cv.resize(img_rgb, cfg.image_dims, interpolation=cv.INTER_AREA)

    roi_crop, _, _ = get_ROI_from_color(img_rgb, cfg)
    tray_no_bg = remove_blue_background(roi_crop, cfg)
    _, bboxes, _ = segment_instruments(tray_no_bg, cfg)
    bbox = _best_tool_bbox(bboxes)
    if bbox is None:
        return None
    # Crop from roi_crop (original colours, blue background intact) to match
    # the inference path in run_pipeline.py, which now also crops from roi_crop.
    return _extract_tool_crop(roi_crop, bbox)


def process_folder(folder: pathlib.Path, cfg: PreprocessConfig) -> tuple[int, int]:
    """Process every image in one Tools/HerramientaN/ folder."""
    dst_folder = DST_ROOT / folder.name
    dst_folder.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        for p in folder.glob(ext)
    )
    ok, fail = 0, 0
    with img_bar(len(image_paths), f"  {folder.name}") as bar:
        for path in image_paths:
            dst = dst_folder / f"{path.stem}.jpg"
            if dst.exists():
                ok += 1
                bar.update(1)
                continue
            try:
                crop = process_training_image(str(path), cfg)
                if crop is None:
                    fail += 1
                else:
                    cv.imwrite(str(dst), cv.cvtColor(crop, cv.COLOR_RGB2BGR))
                    ok += 1
            except Exception as exc:
                print(f"    [fail] {path.name}: {exc}")
                fail += 1
            bar.update(1)
    return ok, fail


def main() -> None:
    if not SRC_ROOT.exists():
        sys.exit(f"Source folder not found: {SRC_ROOT}")

    cfg = PreprocessConfig()

    folders = sorted(
        (f for f in SRC_ROOT.iterdir()
         if f.is_dir() and f.name.startswith("Herramienta")),
        key=lambda p: int(p.name[len("Herramienta"):]),
    )

    print("=" * 60)
    print(f"  Building pipeline crops: {SRC_ROOT} -> {DST_ROOT}")
    print(f"  Classes: {len(folders)}   image_dims: {cfg.image_dims}")
    print("=" * 60)

    t0 = time.time()
    total_ok, total_fail = 0, 0
    for folder in folders:
        ok, fail = process_folder(folder, cfg)
        print(f"  {folder.name:<16s}  ok={ok:>4d}  fail={fail:>3d}")
        total_ok += ok
        total_fail += fail

    elapsed = time.time() - t0
    print()
    print(f"  Total: {total_ok} crops saved, {total_fail} failed "
          f"in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
