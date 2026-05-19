
import numpy as np
from config          import PreprocessConfig
from utils           import load_images
from preprocess      import get_ROI_from_color, binarize_image, get_tray_crop, remove_blue_background
from segmentation    import segment_instruments

def main():
    cfg = PreprocessConfig()

    # -- Image Loading --
    images = load_images("./Trays", cfg)
    img_rgb = images[np.random.randint(0, len(images))]

    # -- Preprocessing --
    roi_crop, roi_mask, roi_bbox = get_ROI_from_color(img_rgb, cfg)
    binary_mask = binarize_image(roi_crop, cfg)
    tray_masked, tray_mask, _ = get_tray_crop(roi_crop, binary_mask, cfg)
    tray_no_bg                = remove_blue_background(tray_masked, cfg)

    # -- Segmentation  --
    seg_binary, bboxes, outlier_analysis, _ = segment_instruments(tray_no_bg, cfg)

if __name__ == "__main__":
    main()
