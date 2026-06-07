# Memory / OOM fix log

Date: 2026-06-07

## Symptom
Running the pipeline / building the classifier intermittently froze the screen or
got the process killed by the Linux OOM killer.

## Root cause
`classifier/classify.py :: load_dataset()` decodes **every** training image into
RAM at once:

```python
images.append(Image.open(path).convert("RGB"))   # held in a list
```

The `Tools/` dataset is **1000 images at ~4032x3024 (some 5712x4284)**.
Decoded to RGB that is ~36 MB each → **~36 GB held simultaneously**, on a machine
with 15 GB RAM + 10 GB swap. This is the OOM / freeze trigger.

The *features* extracted from each image are only 512 floats (~2 KB), and
`build_classifier()` processes one image at a time. There is no need to hold all
decoded images in memory.

## Fix (minimal, scoped to build_classifier only)
Added `classifier/classify.py :: _load_dataset_paths(cfg)` — same as
`load_dataset` but returns image **paths** instead of decoded PIL images.

Rewrote `build_classifier()` to open each image lazily inside the loop, extract
features, then free it. Peak resident image data is now **one** full-res image
(~36 MB) instead of all 1000.

`load_dataset()` itself and the cross-validation functions
(`train_and_evaluate`, `finetune_and_evaluate`) were left untouched to keep the
change small. (They have the same latent problem but are not on the path the user
needs right now; noted here for future work.)

## Operational safeguards used when running
- `MPLBACKEND=Agg` (headless) so no GUI backend can freeze the desktop.
- Thread caps (`OMP_NUM_THREADS` / torch threads) to leave CPU headroom for the
  desktop and avoid the freeze that full 20-core saturation was causing.
- RAM watched while running; abort on dangerous spike.

## Second issue: build never finished (not memory — CPU/time)
After the RAM fix the build ran but, after 10h wall-clock, still had not produced
`cnn_results/`. Diagnosis: it was NOT out of memory (steady ~1.2 GB) and NOT
deadlocked — it was pegged single-threaded on PIL augmentation, having
accumulated only ~46 min of CPU time (the machine was suspended overnight, so
wall-clock ran far ahead of CPU time).

Root cause: `_build_aug_pipeline` applies `RandomRotation` + `ColorJitter` to the
**full-resolution ~12 MP** PIL image, and only afterwards resizes to 224px. With
~4000 augmented copies that is ~40x wasted pixels per op, single-threaded.

Fix: in `build_classifier`, downscale each opened image once to longest-side
640px (`img.thumbnail((640, 640))`) before feature/aug extraction. The network
input is a 224px square regardless, so this is visually equivalent but ~40x
faster. Also added a periodic `[build] N/total` print for visibility (tqdm was
writing nothing to the redirected log).

## Third issue: bad classification accuracy (train/serve skew)
After rebuilding, tray classification confidences were ~26-55% on 10 classes
(near chance). Cause: the classifier was trained on `Tools/` = whole 4032x3024
photos, but at inference it sees tight deskewed crops from `_extract_tool_crop`
(e.g. 320x1889). Completely different distributions.

Fix: generate matched crops with `classifier/build_training_crops.py` ->
`pipeline_crops/`, and point `ClassifierConfig.data_dir` to `./pipeline_crops`.
(`build_training_crops.py` was already in the repo for exactly this but had never
been run, and `data_dir` still pointed at `./Tools`.)

## Fourth issue: Herramienta0 only yielded 30/100 crops
`build_training_crops` only keeps blue-background images. 67/100 of Herramienta0
were shot on a *lighter, less-saturated* blue cloth (H~104, S~21) than the
saturated blue of the inference trays, so the `s>50` saturation cutoff in
`_is_blue_background` rejected them (blue_frac 0.004 vs 0.66 at s>20).

Fix (chosen by user): relax the cutoff `s>50` -> `s>20` in
`_is_blue_background` ONLY (the training-crop script, not the core pipeline).
Re-running regenerates the missing crops (existing ones are skipped via
`dst.exists()`). Note: a few still fail because `get_ROI_from_color` itself is
tuned for saturated blue and raises "No ROI background region detected" on the
lightest images; those are skipped, as before.

## Fifth issue: cross-validation ran >20 min with no result
`train_and_evaluate` (the 5-fold CV) had the same full-size-augmentation
bottleneck the build had, and load_dataset holds all decoded crops (~5.25 GB).
Applied the same downscale-to-640 right after load_dataset (mirrors the
build_classifier fix). This both speeds up augmentation and shrinks the in-RAM
image list. Run with the same OMP thread caps + memory abort guard.

## Sixth issue: Trays2 batch — wrong orientation + over-segmentation
New Trays2 photos (iPhone 16) processed very wrong. Two distinct causes:

1. **EXIF orientation.** Trays2 images are stored 5712x4284 (landscape) with EXIF
   orientation=6 (viewers rotate to portrait). cv.imread ignores EXIF, so the
   pipeline got a sideways image and cv.resize to (4284,5712) distorted it. The
   original Trays/ were stored 4284x5712 with orientation=1. Fix (data, not code):
   baked the orientation into the pixels with PIL `ImageOps.exif_transpose` and
   re-saved all 23 Trays2 images upright (now 4284x5712, orientation normalised).

2. **Over-segmentation (separate).** Even upright, IMG_1261 gave 25 "instruments".
   Cause was NOT the dark margin (get_tray_crop/remove_blue_background are clean) —
   the Sauvola adaptive binariser in segment_instruments amplifies faint cloth-edge
   texture into ~17 tiny (<4k px) false blobs along the image borders. Real tools
   are >=30k px (Trays2) / >=120k px (original Trays), a clear gap. Fix:
   `seg_min_contour_area` 500 -> 10000 in config.py. Validated on 6 Trays2 images
   (counts dropped to a realistic 4-6) and confirmed NO regression on original
   Trays (IMG_3344 still 6 instruments, identical predictions).

## Seventh issue: Trays2/IMG_1271 detected 0 tools
`get_tray_crop` selects the largest contour whose bounding rect is SMALLER than
`tray_full_size_tol` (0.99) of the ROI, to skip a full-frame "desk border"
contour. But when the tray fills the ROI its bounding rect is ~99% too, so the
real tray (area 20.4M px, 99%x99%) was rejected and the only survivors were
noise specks (710 px) -> empty tray mask -> 0 instruments. Other images escaped
this only because their cloth left a margin inside the ROI.

Fix (core, pipeline/preprocess.py): track the largest contour overall and fall
back to it when the filtered candidate is missing or tiny (best_area < 50% of the
largest). The full-size filter still wins when a genuine smaller tray contour
exists. Verified: IMG_1271 0 -> 7 tools; all 23 Trays2 now detect 3-9 tools
(zero empties); original Trays unchanged (IMG_3344/3345 still 6).

## Result
Classifier build peak RAM stays well under 1 GB. Output cached to
`./cnn_results/classifier.joblib` (+ `class_names.json`).
