# run_pipeline.py
# Entry point for running the full instrument segmentation pipeline.
#
# Usage:
#   python run_pipeline.py                    # one random image from ./Trays
#   IMAGE_PATH = "./Trays/IMG_3353.jpg"       # specific image
#   IMAGE_PATH = "all"                        # every image in ./Trays
#
# Pipeline summary:
#   preprocess
#     → segment_instruments      (initial bboxes + outlier classification)
#     → detect_concave_points    (curvature analysis on outlier bboxes)
#     → select_best_concave_points + apply_concave_cuts
#     → find_bboxes              (re-segmentation of the cut mask)
#     → plot_segmentation_results (when debugging=True)

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import cv2 as cv

from config         import PreprocessConfig
from preprocess     import (
    get_ROI_from_color, binarize_image, get_tray_crop,
    remove_blue_background, detect_specular_reflections,
)
from segmentation   import segment_instruments, find_bboxes
from concave_points import detect_concave_points
from concave_cut    import select_best_concave_points, apply_concave_cuts, _concave_grade
from visualize      import plot_segmentation_results, plot_pipeline_result
from classify       import build_classifier, classify_crop, ToolClassifier
from classifier_config import ClassifierConfig
from tool_names    import display_name


def _extract_tool_crop(image_rgb: np.ndarray, bbox: tuple) -> np.ndarray:
    """Deskew the oriented bbox region into an upright RGB crop.

    Orders the four boxPoints as TL/TR/BR/BL, then uses getPerspectiveTransform
    + warpPerspective to produce a straight (w, h) view of the instrument.
    Returns a uint8 RGB array.
    """
    center, size, angle = bbox[0], bbox[1], bbox[2]
    pts = cv.boxPoints(((center[0], center[1]), size, angle)).astype(np.float32)

    s = pts.sum(axis=1)           # x + y
    d = pts[:, 0] - pts[:, 1]    # x - y
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(d)]
    bl = pts[np.argmin(d)]

    w = max(1, int(round(float(np.linalg.norm(tr - tl)))))
    h = max(1, int(round(float(np.linalg.norm(bl - tl)))))

    src = np.array([tl, tr, br, bl], dtype=np.float32)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv.getPerspectiveTransform(src, dst)
    return cv.warpPerspective(image_rgb, M, (w, h))


def _process_image(
    image_path: str,
    cfg: PreprocessConfig,
    debugging: bool = False,
    classifier: ToolClassifier | None = None,
    class_names: list[str] | None = None,
) -> tuple:
    """Run the full pipeline on a single image and optionally display results.

    Pipeline steps:
      1. Load + resize the image.
      2. Preprocessing: ROI crop, blue-background removal, reflection inpaint.
      3. segment_instruments: binarise, find bboxes, classify outliers.
      4. detect_concave_points: curvature analysis on outlier (fused) bboxes.
      5. select_best_concave_points + apply_concave_cuts: cut each fused blob.
      6. find_bboxes on the cut mask → final per-instrument bboxes.
      7. Optional: extract deskewed crops and classify each instrument.
      8. Optionally show the overview and/or classification result figures.

    Args:
        image_path:  Absolute or relative path to a JPEG/PNG image file.
        cfg:         PreprocessConfig controlling all pipeline parameters.
        debugging:   When True, display the visualisation figure(s).
        classifier:  Optional ToolClassifier; when given, each final bbox is
                     classified and the result is included in the return tuple.
        class_names: Ordered list of class names from classifier.class_names.

    Returns:
        9-tuple: (tray_no_bg, seg_binary, seg_binary_cut, final_bboxes,
                  outlier_analysis, concave_pts, crops, pred_labels, confidences)
          crops:       list of deskewed RGB crops (one per final bbox).
          pred_labels: predicted class name strings (empty when no classifier).
          confidences: float confidence scores (empty when no classifier).
    """
    img_bgr = cv.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f'Could not load image: {image_path}')

    img_rgb     = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    img_rgb     = cv.resize(img_rgb, cfg.image_dims, interpolation=cv.INTER_AREA)
    image_label = os.path.basename(image_path)
    print(f'\n[pipeline] {image_path}')

    # ── Preprocessing ────────────────────────────────────────────────────────
    roi_crop, _roi_mask, roi_bbox   = get_ROI_from_color(img_rgb, cfg)
    binary_mask                     = binarize_image(roi_crop, cfg)
    tray_masked, _tray_mask, _      = get_tray_crop(roi_crop, binary_mask, cfg)
    reflection_mask                 = detect_specular_reflections(tray_masked, cfg)
    tray_no_bg                      = remove_blue_background(tray_masked, cfg)

    # ── Core segmentation ────────────────────────────────────────────────────
    seg_binary, bboxes, outlier_analysis = segment_instruments(tray_no_bg, cfg)
    outlier_bboxes = [e['bbox'] for e in outlier_analysis['outliers']]

    # ── Concave-point detection ──────────────────────────────────────────────
    concave_pts = detect_concave_points(seg_binary, outlier_bboxes, cfg)

    # ── Concave cut + re-segmentation ────────────────────────────────────────
    if outlier_bboxes:
        best_points    = select_best_concave_points(concave_pts)
        seg_binary_cut = apply_concave_cuts(
            seg_binary, outlier_bboxes, best_points, cfg,
        )
        final_bboxes   = find_bboxes(seg_binary_cut, cfg)
    else:
        seg_binary_cut = seg_binary
        final_bboxes   = bboxes

    # ── Console summary ──────────────────────────────────────────────────────
    print(f'  Scenario     : {outlier_analysis["scenario"]}')
    print(f'  Initial bboxes : {len(bboxes)}  '
          f'({len(outlier_bboxes)} flagged as fused)')
    print(f'  Final count    : {len(final_bboxes)} instrument(s)')
    for i, bbox in enumerate(final_bboxes):
        center, size, angle, area = bbox
        cx, cy = center
        w, h   = size
        print(f'    #{i + 1}  centre=({cx:.1f}, {cy:.1f})  '
              f'size={w:.1f}×{h:.1f}  angle={angle:.1f}°  area={area:,} px')

    # ── Concave-point summary (★ marks the chosen cut point) ────────────────
    total_cp = sum(len(v) for v in concave_pts.values())
    print(f'  Concave pts  : {total_cp} candidate(s) across '
          f'{len(concave_pts)} outlier bbox(es)')
    if total_cp:
        _hdr = (f"    {'bbox':>4}  {'#':>3}  {'x':>6}  {'y':>6}"
                f"  {'κ':>6}  {'roi_len':>7}  {'chord':>7}  {'grade':>6}  best")
        _sep = '    ' + '─' * (len(_hdr) - 4)
        print(_hdr)
        print(_sep)
        for bidx, pts in concave_pts.items():
            if not pts:
                continue
            grades     = [_concave_grade(p, pts) for p in pts]
            best_grade = max(grades)
            for n, (pt, grade) in enumerate(zip(pts, grades)):
                is_best = '★' if grade == best_grade else ' '
                print(
                    f'    {bidx:>4}  {n:>3}  '
                    f'{pt["x"]:>6}  {pt["y"]:>6}  '
                    f'{pt["kappa"]:>6.3f}  '
                    f'{pt["roi_len"]:>7}  '
                    f'{pt["chord_len"]:>7.1f}  '
                    f'{grade:>6.3f}  {is_best}'
                )

    # ── Classification ───────────────────────────────────────────────────────
    crops: list[np.ndarray] = []
    pred_labels: list[str] = []
    confidences: list[float] = []
    if classifier is not None:
        for bbox in final_bboxes:
            crop = _extract_tool_crop(roi_crop, bbox)
            label_idx, conf = classify_crop(classifier, crop)
            raw = class_names[label_idx] if class_names else str(label_idx)
            name = display_name(raw)
            crops.append(crop)
            pred_labels.append(name)
            confidences.append(conf)
        print(f'  Classifications: '
              + ', '.join(f'{n} ({c:.0%})' for n, c in zip(pred_labels, confidences)))

    # ── Visualisation ────────────────────────────────────────────────────────
    if debugging:
        plot_segmentation_results(
            tray_masked, tray_no_bg, seg_binary, bboxes,
            reflection_mask=reflection_mask,
            image_label=image_label,
            outlier_analysis=outlier_analysis,
            img_rgb=img_rgb,
            roi_bbox=roi_bbox,
            cfg=cfg,
            concave_points=concave_pts  if outlier_bboxes else None,
            seg_binary_cut=seg_binary_cut if outlier_bboxes else None,
            final_bboxes=final_bboxes   if outlier_bboxes else None,
        )
        if classifier is not None:
            plot_pipeline_result(
                tray_masked, final_bboxes, crops, pred_labels, confidences,
                image_label=image_label,
                save_path=f'./pipeline_results/{image_label}_result.png',
            )

    return (tray_no_bg, seg_binary, seg_binary_cut, final_bboxes,
            outlier_analysis, concave_pts, crops, pred_labels, confidences)


def main(
    debugging: bool = False,
    image_path: str | None = None,
    classify: bool = False,
) -> object:
    """Run the pipeline on one image (random if None) or all images in ./Trays.

    Args:
        debugging:   Show visualisation figures after processing each image.
        image_path:  Path to a specific image, None for a random image,
                     or 'all' to process every image found in ./Trays.
        classify:    When True, build a ToolClassifier once and classify every
                     detected instrument crop.  Results are included in the
                     returned tuple(s) and a figure is saved to ./pipeline_results/.

    Returns:
        Single 9-tuple from _process_image, or a list of them when image_path='all'.
    """
    cfg = PreprocessConfig()

    classifier = None
    class_names = None
    if classify:
        print('[pipeline] Building classifier (runs once)…')
        classifier = build_classifier(ClassifierConfig())
        class_names = classifier.class_names

    def _collect_paths() -> list[str]:
        paths = sorted(
            os.path.join(wd, f)
            for wd, _, files in os.walk('./Trays')
            for f in files
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
        )
        if not paths:
            raise FileNotFoundError('No images found in ./Trays')
        return paths

    if image_path == 'all':
        all_paths = _collect_paths()
        print(f'Processing {len(all_paths)} image(s)…')
        results = []
        for path in all_paths:
            try:
                results.append(_process_image(
                    path, cfg, debugging=debugging,
                    classifier=classifier, class_names=class_names,
                ))
            except Exception as exc:
                print(f'  [error] {path}: {exc}')
        return results

    if image_path is None:
        all_paths  = _collect_paths()
        image_path = all_paths[np.random.randint(0, len(all_paths))]

    return _process_image(
        image_path, cfg, debugging=debugging,
        classifier=classifier, class_names=class_names,
    )


if __name__ == '__main__':
    IMAGE_PATH = 'all'   # None = random · 'all' = every image · or a specific path
    main(debugging=True, image_path=IMAGE_PATH, classify=True)
