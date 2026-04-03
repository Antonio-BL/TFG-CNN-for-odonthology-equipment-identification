
import numpy as np
from config          import PreprocessConfig
from utils           import load_images
from preprocess      import get_ROI_from_color, binarize_image
from segmentation    import segment_instruments

def main():
    cfg = PreprocessConfig()

    # -- Image Loading --
    # debug=True (set in PreprocessConfig): loads one random image.
    # debug=False: loads all images, using a disk cache at ./cache/images.pkl.
    images = load_images("./Trays", cfg)
    img_rgb = images[np.random.randint(0, len(images))]

    # -- Preprocessing --
    roi_crop, roi_mask, roi_bbox = get_ROI_from_color(img_rgb, cfg)
    binary_mask = binarize_image(roi_crop, cfg)

    # -- Segmentation  --
    contours = segment_instruments(roi_crop, binary_mask, cfg)

if __name__ == "__main__":
    main()