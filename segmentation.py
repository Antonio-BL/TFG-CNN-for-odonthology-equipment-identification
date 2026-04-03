# segmentation.py
# Segments the tray ROI after preprocessing.
# 1. Extract contours from the binary mask.
# 2. Compute oriented bounding boxes and geometric/photometric features.
# 3. Flag size-based outlier regions.
# 4. Visualise results with matplotlib.

# ================================================================== #
# Basic dependencies                                                 #
# ================================================================== #
import os
import platform
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dataclasses import dataclass, field

# ================================================================== #
# Linux support                                                      #
# ================================================================== #
if platform.system() == "Linux":
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

# ================================================================== #
# Project imports                                                    #
# ================================================================== #
from config     import PreprocessConfig
from utils      import (load_images, get_multi_patches, get_avg_color)
from preprocess import get_ROI_from_color, binarize_image

# ================================================================== #
# Data structures                                                    #
# ================================================================== #

@dataclass
class InstrumentData:
    """
    Holds the contour and all computed features for one detected instrument
    region. Fields with default values are safe to omit when constructing
    from outside this module (existing call sites will not break).
    """
    contour:               np.ndarray                                        # raw contour points from cv.findContours
    center:                tuple                                             # (cx, cy) float pixel coords
    size:                  tuple                                             # (long_side, short_side) in pixels
    angle:                 float       = 0.0                                 # compass bearing of long axis (0=up, 90=right)
    aspect_ratio:          float       = 1.0
    solidity:              float       = 1.0
    is_outlier:            bool        = False
    extent:                float       = 0.0
    hu_moments:            np.ndarray  = field(default_factory=lambda: np.zeros(7))
    symmetry_score:        float       = 0.0
    defect_depth_ratio:    float       = 0.0
    n_significant_defects: int         = 0
    mass_asymmetry:        float       = 1.0


# ================================================================== #
# Feature extraction functions                                       #
# ================================================================== #

def compute_oriented_bbox(contour: np.ndarray):
    """
    Computes the minimum-area oriented bounding box of a contour and returns
    a normalised (center, size, angle) tuple.

    OpenCV's minAreaRect angle convention
    --------------------------------------
    cv.minAreaRect returns an angle in (-90, 0] that describes the clockwise
    rotation of the rectangle's *width* edge from the positive x-axis.
    Crucially, OpenCV chooses which pair of sides to label "width" and
    "height" solely to keep the angle inside (-90, 0] — not to ensure width
    is the longer side.  For a tall, narrow tool (height > width in the
    natural sense), OpenCV labels the *short* horizontal edge as the "width",
    so the raw angle still describes that short edge and the long axis must be
    inferred by adding 90°.

    This function corrects for that ambiguity:
      - long_side >= short_side is always guaranteed.
      - The returned angle is converted to a compass bearing (0° = north/up,
        90° = east/right) so that the visualisation formula
        cx + (L/2)*sin(angle_rad), cy − (L/2)*cos(angle_rad) points in the
        direction of the long axis.

    Args:
        contour: np.ndarray of contour points (output of cv.findContours).

    Returns:
        center: (cx, cy) float centre of the bounding box.
        size:   (long_side, short_side) in pixels, long_side >= short_side.
        angle:  compass bearing of the long axis in degrees, in [0, 180).
    """
    (cx, cy), (w, h), raw_angle = cv.minAreaRect(contour)

    # Determine which OpenCV dimension is the long axis and find the angle
    # of that long axis measured from horizontal (clockwise in image coords).
    if w >= h:
        long_side, short_side = w, h
        long_from_horiz = raw_angle          # width IS the long side
    else:
        long_side, short_side = h, w
        long_from_horiz = raw_angle + 90.0   # long axis is 90° from the short "width" edge

    # Convert from "degrees clockwise from east" to compass bearing
    # (degrees clockwise from north).  In image coords (y pointing down):
    #   compass = 90 − long_from_horiz
    angle = (90.0 - long_from_horiz) % 180.0

    return (cx, cy), (long_side, short_side), angle


def compute_aspect_ratio(size: tuple) -> float:
    """
    Computes the aspect ratio of an oriented bounding box.

    Aspect ratio = long_side / short_side.  A single elongated instrument
    (clamp, retractor, scissors) typically has aspect ratio > 3.  A roughly
    square or compact blob (or two tools overlapping end-to-end) has a lower
    ratio.

    Args:
        size: (long_side, short_side) tuple from compute_oriented_bbox.

    Returns:
        Aspect ratio >= 1.0.  Returns 1.0 if short_side is near zero to
        guard against division by zero.
    """
    long_side, short_side = size
    if short_side < 1e-6:
        return 1.0
    return long_side / short_side


def compute_solidity(contour: np.ndarray) -> float:
    """
    Computes the solidity of a contour: contour_area / convex_hull_area.

    The convex hull is the smallest convex polygon that encloses the contour
    — imagine stretching a rubber band around the silhouette.  A convex shape
    (circle, rectangle) has solidity ≈ 1.0.  When two tools touch, their
    joined silhouette develops concave "bites" at the contact point; the hull
    must bridge those indentations, enlarging hull_area while contour_area
    stays the same, so solidity drops noticeably below 1.0.

    Args:
        contour: np.ndarray of contour points.

    Returns:
        Solidity in [0, 1].  Returns 0.0 if the hull area is zero.
    """
    contour_area = cv.contourArea(contour)
    hull         = cv.convexHull(contour)
    hull_area    = cv.contourArea(hull)
    if hull_area < 1e-6:
        return 0.0
    return contour_area / hull_area


def compute_extent(contour: np.ndarray) -> float:
    """
    Computes the extent of a contour: contour_area / axis_aligned_bbox_area.

    The denominator is the area of cv.boundingRect — the smallest upright
    (non-rotated) rectangle that encloses the contour.

    Extent vs. solidity
    -------------------
    Solidity uses the convex hull, which rotates to fit the tool tightly.
    Extent uses the axis-aligned box, which does not rotate.  A thin diagonal
    tool has *low* extent (much of its upright box is empty background) but
    *high* solidity (the hull follows the tool closely).  Extent is therefore
    more sensitive to how well the tool's orientation aligns with the image
    axes and provides a complementary signal to solidity.

    Args:
        contour: np.ndarray of contour points.

    Returns:
        Extent in (0, 1].  Returns 0.0 if the bounding rectangle area is zero.
    """
    contour_area = cv.contourArea(contour)
    _, _, w, h   = cv.boundingRect(contour)
    bbox_area    = float(w * h)
    if bbox_area < 1e-6:
        return 0.0
    return contour_area / bbox_area


def compute_hu_moments(contour: np.ndarray) -> np.ndarray:
    """
    Computes the seven Hu moments of a contour with log-compression.

    Hu moments are seven scalars derived from the spatial moments of a binary
    shape.  They are invariant to translation, scale, and rotation (and the
    seventh is additionally invariant to reflection).  They capture global
    shape properties such as elongation and bilateral symmetry.

    Why log-compression is necessary
    ---------------------------------
    Raw Hu moments span many orders of magnitude within a single shape
    (values range from roughly 1e-2 down to 1e-20).  Feeding them directly
    into any distance-based computation or normalisation would let the
    first moment completely dominate.  The standard compression:

        h_compressed = sign(h) * log10(|h| + ε),   ε = 1e-10

    maps the span [1e-20, 1] to roughly [−10, 0] while preserving the sign
    and relative ordering of each moment.  The ε floor prevents log10(0) = −∞.

    Args:
        contour: np.ndarray of contour points.

    Returns:
        np.ndarray of shape (7,) containing the log-compressed Hu moments.
    """
    moments = cv.moments(contour)
    hu      = cv.HuMoments(moments).flatten()          # shape (7,)
    return np.sign(hu) * np.log10(np.abs(hu) + 1e-10)


def compute_axial_symmetry(contour: np.ndarray, binary_mask: np.ndarray) -> float:
    """
    Estimates the bilateral (left–right) symmetry of a contour region by
    rotating its binary mask crop so the long axis is vertical, then
    computing IoU between the left half and the mirrored right half.

    Physical interpretation
    -----------------------
    A single elongated instrument (clamp, retractor, scissors) is
    approximately bilaterally symmetric along its long axis → IoU near 1.0.
    Two tools touching side by side break this symmetry: one half of the
    aligned bounding box contains more mass than the other → IoU near 0.0.

    Algorithm
    ---------
    1.  Compute the axis-aligned bounding box (cv.boundingRect) of the
        contour and crop binary_mask with a small padding border to avoid
        clipping the silhouette after rotation.
    2.  Get the compass angle of the long axis from compute_oriented_bbox.
    3.  Rotate the crop so the long axis points upward using
        cv.getRotationMatrix2D (angle = −compass_angle) and cv.warpAffine.
        The negation converts from the compass convention (CW from north) to
        OpenCV's rotation convention (CCW in image-display coordinates).
    4.  Split the rotated crop into left and right halves at the centre column.
    5.  Flip the right half horizontally with cv.flip(right, flipCode=1).
    6.  Trim both halves to equal width to handle odd total widths.
    7.  Compute IoU = |left ∩ right_flipped| / (|left ∪ right_flipped| + 1e-6)
        where ∩ and ∪ are pixel-wise (cv.bitwise_and / cv.bitwise_or) and
        |·| counts non-zero pixels.

    Args:
        contour:     np.ndarray of contour points.
        binary_mask: uint8 binary mask (tools = 255, background = 0), same
                     spatial extent as the ROI crop passed to binarize_image.

    Returns:
        IoU score in [0.0, 1.0].  Near 1.0 = symmetric; near 0.0 = asymmetric.
    """
    pad = 5
    x, y, w, h    = cv.boundingRect(contour)
    H_mask, W_mask = binary_mask.shape[:2]
    x0 = max(0, x - pad);  x1 = min(W_mask, x + w + pad)
    y0 = max(0, y - pad);  y1 = min(H_mask, y + h + pad)
    crop = binary_mask[y0:y1, x0:x1].copy()

    if crop.size == 0:
        return 0.0

    _, _, angle = compute_oriented_bbox(contour)

    cH, cW = crop.shape[:2]
    center  = (cW / 2.0, cH / 2.0)
    # Negate angle: compass convention (CW from north) → OpenCV rotation
    # convention (positive = CCW in image display coords, i.e. CCW in standard
    # math with y-axis flipped, which is CW visually on screen — hence -angle
    # rotates the content so that the long axis ends up pointing upward).
    M       = cv.getRotationMatrix2D(center, -angle, 1.0)
    rotated = cv.warpAffine(crop, M, (cW, cH),
                            flags=cv.INTER_NEAREST,
                            borderMode=cv.BORDER_CONSTANT, borderValue=0)

    mid        = rotated.shape[1] // 2
    left       = rotated[:, :mid]
    right_flip = cv.flip(rotated[:, mid:], 1)          # mirror right half

    # Trim to equal width (handles odd total column counts)
    min_w      = min(left.shape[1], right_flip.shape[1])
    left       = left[:, :min_w]
    right_flip = right_flip[:, :min_w]

    intersection = cv.bitwise_and(left, right_flip)
    union        = cv.bitwise_or(left,  right_flip)
    iou = np.count_nonzero(intersection) / (np.count_nonzero(union) + 1e-6)
    return float(iou)


def compute_convexity_defect_features(contour: np.ndarray) -> tuple:
    """
    Summarises the convexity defects of a contour as two scalars: the
    normalised depth of the deepest defect and the count of significant ones.

    A convexity defect is a region of the contour that dents inward relative
    to the convex hull.  cv.convexityDefects reports each defect as a 4-tuple
    (start_idx, end_idx, farthest_idx, depth×256); dividing depth by 256
    gives the true pixel depth.

    Physical interpretation
    -----------------------
    - Two touching tools: one deep, narrow indentation at the contact point
      → large depth_ratio, n_significant_defects ≥ 1.
    - Forceps / tweezers handle: one long, shallow channel running along the
      length → small depth_ratio, n_significant_defects = 0 or 1.
    - Solid single tool (clamp body, retractor paddle): nearly convex
      → depth_ratio ≈ 0, n_significant_defects = 0.

    Args:
        contour: np.ndarray of contour points.

    Returns:
        depth_ratio:          deepest defect depth / contour perimeter.
                              0.0 if no defects exist.
        n_significant_defects: count of defects with depth > 0.02 × perimeter.
    """
    perimeter = cv.arcLength(contour, True)
    if perimeter < 1e-6:
        return 0.0, 0

    hull_idx = cv.convexHull(contour, returnPoints=False)
    if hull_idx is None or len(hull_idx) < 3:
        return 0.0, 0

    defects = cv.convexityDefects(contour, hull_idx)
    if defects is None:
        return 0.0, 0

    # Column 3 stores depth × 256 as a fixed-point integer
    depths             = defects[:, 0, 3].astype(np.float64) / 256.0
    threshold          = 0.02 * perimeter
    depth_ratio        = float(depths.max() / perimeter)
    n_significant      = int(np.sum(depths > threshold))
    return depth_ratio, n_significant


def compute_mass_asymmetry(contour: np.ndarray, binary_mask: np.ndarray) -> float:
    """
    Measures the pixel-mass imbalance between the two halves of a contour
    region when split along the long axis.

    The crop of binary_mask around the contour is rotated so the long axis
    is vertical (using the same rotation as compute_axial_symmetry).  The
    rotated crop is split at the centre column: left_sum and right_sum are
    the total foreground pixel intensity on each side.  The ratio
    left_sum / (right_sum + 1e-6) is returned.

    Interpretation: a value near 1.0 indicates balanced mass (expected for
    a single symmetric tool).  A value substantially above or below 1.0
    indicates a skewed silhouette, consistent with two tools whose centres
    do not overlap along the long axis.

    Args:
        contour:     np.ndarray of contour points.
        binary_mask: uint8 binary mask, same spatial extent as the ROI crop.

    Returns:
        left_sum / (right_sum + 1e-6) as a float.
    """
    pad = 5
    x, y, w, h    = cv.boundingRect(contour)
    H_mask, W_mask = binary_mask.shape[:2]
    x0 = max(0, x - pad);  x1 = min(W_mask, x + w + pad)
    y0 = max(0, y - pad);  y1 = min(H_mask, y + h + pad)
    crop = binary_mask[y0:y1, x0:x1].copy()

    if crop.size == 0:
        return 1.0

    _, _, angle = compute_oriented_bbox(contour)

    cH, cW = crop.shape[:2]
    center  = (cW / 2.0, cH / 2.0)
    M       = cv.getRotationMatrix2D(center, -angle, 1.0)   # see compute_axial_symmetry
    rotated = cv.warpAffine(crop, M, (cW, cH),
                            flags=cv.INTER_NEAREST,
                            borderMode=cv.BORDER_CONSTANT, borderValue=0)

    mid       = rotated.shape[1] // 2
    left_sum  = float(np.sum(rotated[:, :mid].astype(np.float64)))
    right_sum = float(np.sum(rotated[:, mid:].astype(np.float64)))
    return left_sum / (right_sum + 1e-6)


def extract_all_features(contour: np.ndarray, binary_mask: np.ndarray) -> dict:
    """
    Convenience wrapper that calls every feature-extraction function for one
    contour and returns the results as a flat dict.

    This dict is the canonical feature record for one detected instrument
    region.  It populates InstrumentData and will later serve as the source
    for building an SVM feature vector — that vector is not built here.

    Keys
    ----
    center, size, angle       from compute_oriented_bbox
    aspect_ratio              from compute_aspect_ratio
    solidity                  from compute_solidity
    extent                    from compute_extent
    hu_moments                np.ndarray (7,) from compute_hu_moments
    symmetry_score            from compute_axial_symmetry
    defect_depth_ratio        from compute_convexity_defect_features
    n_significant_defects     from compute_convexity_defect_features
    mass_asymmetry            from compute_mass_asymmetry

    Args:
        contour:     np.ndarray of contour points.
        binary_mask: uint8 binary mask for the full ROI crop.

    Returns:
        dict with the keys listed above.
    """
    center, size, angle        = compute_oriented_bbox(contour)
    aspect_ratio               = compute_aspect_ratio(size)
    solidity                   = compute_solidity(contour)
    extent                     = compute_extent(contour)
    hu_moments                 = compute_hu_moments(contour)
    symmetry_score             = compute_axial_symmetry(contour, binary_mask)
    depth_ratio, n_sig         = compute_convexity_defect_features(contour)
    mass_asymmetry             = compute_mass_asymmetry(contour, binary_mask)

    return dict(
        center                = center,
        size                  = size,
        angle                 = angle,
        aspect_ratio          = aspect_ratio,
        solidity              = solidity,
        extent                = extent,
        hu_moments            = hu_moments,
        symmetry_score        = symmetry_score,
        defect_depth_ratio    = depth_ratio,
        n_significant_defects = n_sig,
        mass_asymmetry        = mass_asymmetry,
    )


# ================================================================== #
# Contour extraction and segmentation                                #
# ================================================================== #

_MIN_CONTOUR_AREA = 500   # px² — filters microscopic noise at full resolution


def extract_contours(
    roi_crop:    np.ndarray,
    binary_mask: np.ndarray,
    cfg:         PreprocessConfig,
    min_area:    int = _MIN_CONTOUR_AREA,
) -> list:
    """
    Finds external contours in binary_mask, filters by minimum area, and
    returns one fully-populated InstrumentData instance per valid region.

    Args:
        roi_crop:    RGB image of the tray ROI (carried through for context).
        binary_mask: uint8 binary mask (tools = 255, background = 0).
        cfg:         PreprocessConfig (reserved for future per-config thresholds).
        min_area:    Minimum contour area in pixels² to be considered valid.
                     Defaults to _MIN_CONTOUR_AREA.

    Returns:
        List of InstrumentData, one per detected region, with all feature
        fields populated by extract_all_features.
    """
    contours, _ = cv.findContours(binary_mask, cv.RETR_EXTERNAL,
                                  cv.CHAIN_APPROX_SIMPLE)
    instruments = []
    for cnt in contours:
        if cv.contourArea(cnt) < min_area:
            continue
        feats = extract_all_features(cnt, binary_mask)
        inst  = InstrumentData(
            contour               = cnt,
            center                = feats['center'],
            size                  = feats['size'],
            angle                 = feats['angle'],
            aspect_ratio          = feats['aspect_ratio'],
            solidity              = feats['solidity'],
            extent                = feats['extent'],
            hu_moments            = feats['hu_moments'],
            symmetry_score        = feats['symmetry_score'],
            defect_depth_ratio    = feats['defect_depth_ratio'],
            n_significant_defects = feats['n_significant_defects'],
            mass_asymmetry        = feats['mass_asymmetry'],
        )
        instruments.append(inst)
    return instruments


def segment_instruments(
    roi_crop:    np.ndarray,
    binary_mask: np.ndarray,
    cfg:         PreprocessConfig,
) -> list:
    """
    Main segmentation entry point.  Extracts contours, computes all features,
    and flags size-based outliers.

    Outlier detection: a region is marked as is_outlier = True if its area
    deviates from the population mean by more than two standard deviations.
    This catches merged blobs (unusually large) and residual noise that
    survived morphological cleanup (unusually small).

    Args:
        roi_crop:    RGB image of the tray ROI.
        binary_mask: uint8 binary mask (tools = 255, background = 0).
        cfg:         PreprocessConfig.

    Returns:
        List of InstrumentData instances with is_outlier set appropriately.
    """
    instruments = extract_contours(roi_crop, binary_mask, cfg)

    if len(instruments) > 1:
        areas     = np.array([cv.contourArea(i.contour) for i in instruments],
                             dtype=np.float64)
        mean_area = areas.mean()
        std_area  = areas.std()
        for inst, area in zip(instruments, areas):
            if abs(area - mean_area) > 2.0 * std_area:
                inst.is_outlier = True

    return instruments


# ================================================================== #
# Visualisation                                                      #
# ================================================================== #

def visualise_results(
    roi_crop:    np.ndarray,
    binary_mask: np.ndarray,
    instruments: list,
) -> None:
    """
    Displays a 2×2 matplotlib figure summarising the segmentation output.

    Subplots
    --------
    [0,0] Binary mask        — raw binarized mask, greyscale.
    [0,1] Masked ROI         — roi_crop pixels visible only where the mask is
                               255; black elsewhere.  Constructed with NumPy
                               indexing, no OpenCV drawing calls.
    [1,0] Oriented bboxes    — matplotlib Polygon patches drawn over roi_crop.
                               Normal: limegreen.  Outlier: red.
    [1,1] Tool orientations  — annotate arrows from each contour centre toward
                               the tip of the long axis.
                               Normal: green.  Outlier: red.

    All subplots have axis('off').  A suptitle reports the total region count.

    Args:
        roi_crop:    RGB image of the tray ROI.
        binary_mask: uint8 binary mask (tools = 255, background = 0).
        instruments: list of InstrumentData from segment_instruments.
    """
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    # -- [0,0] Binary mask --------------------------------------------------
    axs[0, 0].imshow(binary_mask, cmap='gray')
    axs[0, 0].set_title("Binary mask")
    axs[0, 0].axis('off')

    # -- [0,1] Masked ROI (NumPy indexing only, no OpenCV calls) ------------
    masked = np.zeros_like(roi_crop)
    masked[binary_mask == 255] = roi_crop[binary_mask == 255]
    axs[0, 1].imshow(masked)
    axs[0, 1].set_title("Masked ROI")
    axs[0, 1].axis('off')

    # -- [1,0] Oriented bounding boxes ---------------------------------------
    axs[1, 0].imshow(roi_crop)
    for inst in instruments:
        corners = cv.boxPoints(cv.minAreaRect(inst.contour))   # shape (4,2)
        color   = 'red' if inst.is_outlier else 'limegreen'
        poly    = mpatches.Polygon(corners, closed=True, fill=False,
                                   edgecolor=color, linewidth=1.5)
        axs[1, 0].add_patch(poly)
    axs[1, 0].set_title("Oriented bounding boxes")
    axs[1, 0].axis('off')

    # -- [1,1] Tool orientations (compass arrows) ----------------------------
    axs[1, 1].imshow(roi_crop)
    for inst in instruments:
        cx, cy       = inst.center
        long_side, _ = inst.size
        angle_rad    = np.radians(inst.angle)
        # Arrow endpoint along the long-axis direction.
        # Formula: angle=0 → north (up), angle=90 → east (right).
        ex = cx + (long_side / 2.0) * np.sin(angle_rad)
        ey = cy - (long_side / 2.0) * np.cos(angle_rad)
        color = 'red' if inst.is_outlier else 'green'
        axs[1, 1].annotate(
            '',
            xy=(ex, ey), xytext=(cx, cy),
            arrowprops=dict(arrowstyle='->', color=color, lw=1.5),
        )
    axs[1, 1].set_title("Tool orientations")
    axs[1, 1].axis('off')

    n = len(instruments)
    plt.suptitle(f"Segmentation results — {n} instrument region(s) detected")
    plt.tight_layout()
    plt.show()


# ================================================================== #
# Main                                                               #
# ================================================================== #

def main(debugging: bool = False):
    """
    Standalone entry point for segmentation.py.  Loads one image, runs the
    full preprocessing and segmentation pipeline, and optionally displays the
    results via visualise_results.

    Args:
        debugging: When True, calls visualise_results after segmentation.
    """
    cfg = PreprocessConfig()

    # -- Load images --
    tray_images = load_images("./Trays", cfg)
    if not tray_images:
        raise FileNotFoundError("No images found in ./Trays")

    img_rgb = tray_images[np.random.randint(0, len(tray_images))]

    # -- Preprocessing --
    roi_crop, roi_mask, roi_bbox = get_ROI_from_color(img_rgb, cfg)
    binary_mask = binarize_image(roi_crop, cfg)

    # -- Segmentation --
    instruments = segment_instruments(roi_crop, binary_mask, cfg)

    # -- Debug visualisation --
    if debugging:
        visualise_results(roi_crop, binary_mask, instruments)

    return roi_crop, binary_mask, instruments


if __name__ == "__main__":
    main(debugging=True)
