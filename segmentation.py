# Segments images after preprocessing.

# ================================================================== #
# Basic dependencies                                                 #
# ================================================================== #
import os
import platform
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# ================================================================== #
# Linux support                                                      #
# ================================================================== #
if platform.system() == "Linux":
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

# ================================================================== #
# Project imports                                                    #
# ================================================================== #
from config        import PreprocessConfig
from utils         import (load_images, get_multi_patches, get_avg_color)
from preprocess    import get_ROI_from_color, binarize_image

# ================================================================== #
# Local functions                                                    #
# ================================================================== #
# (your segmentation functions will go here)

# ================================================================== #
# Main                                                               #
# ================================================================== #
def main(debugging: bool = False):
    cfg = PreprocessConfig()

    # -- Load images --
    tray_images = load_images("./Trays", cfg)
    if not tray_images:
        raise FileNotFoundError("No images found in ./Trays")

    img_rgb = tray_images[np.random.randint(0, len(tray_images))]

    # -- Preprocessing --
    roi_crop, roi_mask, roi_bbox = get_ROI_from_color(img_rgb, cfg)
    roi_masked_image = cv.bitwise_and(img_rgb, img_rgb, mask=roi_mask)

    binary_mask = binarize_image(roi_crop, cfg)

    # -- Debug visualizations --
    if debugging:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        axs[0].imshow(img_rgb)
        axs[0].set_title("Original")
        axs[0].axis("off")

        axs[1].imshow(roi_crop)
        axs[1].set_title("ROI crop")
        axs[1].axis("off")

        axs[2].imshow(binary_mask, cmap="gray")
        axs[2].set_title("Binary mask")
        axs[2].axis("off")

        plt.tight_layout()
        plt.show()

    return roi_crop, binary_mask


if __name__ == "__main__":
    main(debugging=True)