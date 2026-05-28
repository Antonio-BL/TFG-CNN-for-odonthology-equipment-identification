# Segmentation Benchmark Report

**Generated:** 2026-05-16 17:55  
**Images processed:** 21  
**Trays directory:** `./Trays`

## Overview

Compares two segmentation pipelines on every image in the Trays dataset:

| Pipeline | Description |
|---|---|
| **With KNN** (`segmentation.py`) | morphological close → Otsu → KNN/convex-hull clustering → bbox filter → outlier analysis |
| **Without KNN** (`segmentation_test.py`) | morphological close → Otsu → bbox filter → outlier analysis |

## Summary Statistics

| Metric | With KNN | Without KNN | Δ (KNN − base) |
|---|---|---|---|
| Mean bboxes/image | 6.0 | 6.0 | +0.0 |
| Min bboxes        | 5 | 5 | — |
| Max bboxes        | 8 | 8 | — |
| Images where KNN **reduced** bbox count (merging)  | — | — | 0/21 |
| Images where KNN **increased** bbox count (splitting) | — | — | 0/21 |
| Images **unchanged** | — | — | 21/21 |

## Per-Image Results

| Image | With KNN (bboxes) | Outlier scenario | Without KNN (bboxes) | Outlier scenario | Δ | Notes |
|---|---|---|---|---|---|---|
| `IMG_3344.jpg` | 6 | none | 6 | single | 0 |  |
| `IMG_3345.jpg` | 6 | none | 6 | single | 0 |  |
| `IMG_3346.jpg` | 5 | none | 5 | multiple | 0 |  |
| `IMG_3347.jpg` | 5 | none | 5 | none | 0 |  |
| `IMG_3348.jpg` | 5 | none | 5 | none | 0 |  |
| `IMG_3349.jpg` | 6 | none | 6 | single | 0 |  |
| `IMG_3350.jpg` | 6 | none | 6 | single | 0 |  |
| `IMG_3351.jpg` | 6 | none | 6 | single | 0 |  |
| `IMG_3353.jpg` | 5 | none | 5 | single | 0 |  |
| `IMG_3354.jpg` | 6 | none | 6 | single | 0 |  |
| `IMG_3355.jpg` | 5 | none | 5 | none | 0 |  |
| `IMG_3356.jpg` | 6 | none | 6 | single | 0 |  |
| `IMG_3357.jpg` | 7 | none | 7 | multiple | 0 |  |
| `IMG_3359.jpg` | 6 | none | 6 | multiple | 0 |  |
| `IMG_3360.jpg` | 6 | none | 6 | none | 0 |  |
| `IMG_3361.jpg` | 6 | none | 6 | none | 0 |  |
| `IMG_3365.jpg` | 6 | none | 6 | none | 0 |  |
| `IMG_3368.jpg` | 5 | none | 5 | single | 0 |  |
| `IMG_3373.jpg` | 8 | none | 8 | single | 0 |  |
| `IMG_3374.jpg` | 8 | none | 8 | multiple | 0 |  |
| `IMG_3376.jpg` | 7 | none | 7 | none | 0 |  |

## Interpretation Notes

- **Δ < 0**: KNN clustering merged contour fragments into fewer, larger bounding boxes (desirable when instrument parts were split across multiple contours).
- **Δ > 0**: KNN clustering introduced extra bounding boxes, possibly by expanding convex hulls over unrelated regions.
- **Δ = 0**: Both pipelines agree on the instrument count for this image.
- **Outlier scenario `single`**: one bbox is ≥2× the median area — likely two touching instruments fused into one contour.
- **Outlier scenario `multiple`**: two or more such fused regions detected.

