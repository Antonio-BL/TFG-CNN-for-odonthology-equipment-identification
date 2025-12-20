# This progrram tes the files containing trays with multiple tools and extracts all 
# the tools as iividual images, that will be then given to CNN for classification.
# The tools usedill be classical segmnentation tools, without using any ML or DL methods. 

# cfg parameters may be passed by some other script or webapp

# %% Dependencies Import

# ----------------------------------------- #
# Basic dependencies                        #
# ----------------------------------------- #
import os
import platform 
from typing import Optional, Iterable

import numpy as np 
import pandas as pd 

import seaborn as sns
import matplotlib.pyplot as plt

from dataclasses import dataclass
# ----------------------------------------- #
# Image processing dependencies             #
# ----------------------------------------- #
import cv2 as cv
from scipy.signal import convolve2d

# ----------------------------------------- #
# Linux support                             #
# ----------------------------------------- #
# force x11
if platform.system() == "Linux":
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

# ----------------------------------------- #
#  Preprocess Configuration class           # 
# ----------------------------------------- #
@dataclass
class PreprocessConfig:
    image_dims: tuple[int, int] = (244, 244)
    color_filter_tolerance: float = 0.5
    crop_filter_tolerance: float = 0.17
    open_kernel_dims: tuple[int, int]= (3, 3)
    close_kernel_dims: tuple[int, int]= (20, 20)
    patch_center: int = None
    patch_size: int = 10

# ----------------------------------------- #
# Helper functions                          #
# ----------------------------------------- #

# Loads images
def load_images(path: str, cfg: PreprocessConfig) -> list[np.ndarray]:
    '''
    Reads images in specified directories and loads them in an array.
    Args: 
        path: Path to scan for images.
        cfg: Configuration parameters.
    Returns: 
        Array containing scanned images.
    Raises: 
        FileNotFoundError: if the directory does not exist
    ''' 
    if not os.path.isdir(path): 
        raise FileNotFoundError(f"{path} is not a directory \n")

    images = []
    for working_directory, directory, file in os.walk(path):
        for f in file:
            img_path = os.path.join(working_directory, f)
            img= cv.imread(img_path)
            if img is None:
                print(f" Warning: Image {img_path} has been skipped: It is empty \n")
                continue
         
            img = cv.resize(img, cfg.image_dims, interpolation=cv.INTER_AREA)
            images.append(img)
               
    return images

# ----------------------------------------- #
# Image manipulation functions              #a: int =
# ----------------------------------------- #
def get_centered_patch(image: np.ndarray, cfg: PreprocessConfig) -> dict:
    '''
    Calculates de center area within a RGB image.
    Args: 
        image: RGB image.
        cfg: configuration Parameters
    Returns: 
        Dictionary containing the area data.
            - patch: np.ndarray containing the area.
            - vortex0: one vortex of the area.
            - vortex1: opposite vortex to vortex0.
    '''
    
    [img_height, img_width ] = np.shape(image)[0:2]

    if cfg.patch_center is None:
        center = np.array([img_width // 2, img_height // 2], dtype=int)  # (x, y)
    else:
        center = np.array(cfg.patch_center, dtype=int)
   
    # define area to get color as 10% 
    area_size = cfg.patch_size / 100; 
    area = (np.array(np.multiply([img_width, img_height], area_size))).astype(int)
    
    # get averages of each R G B color for the image
    x0 = max(center[0] - area[0] // 2, 0)
    x1 = min(center[0] + area[0] // 2, img_width)
    y0 = max(center[1] - area[1] // 2, 0)
    y1 = min(center[1] + area[1] // 2, img_height)

    img_patch = image[y0:y1, x0:x1]

    area_data = {
        "patch" : np.array(img_patch),
        "vortex0" : np.array([x0,y0]).astype(int),
        "vortex1" : np.array([x1,y1]).astype(int), 
    }
    return area_data


def get_avg_color(colored_area: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    '''
    Gets the average color of an image. Using a custom method.
    Args: 
        colored_area: image to get the average color of.
        cfg: Configuration parameters
    Returns: 
        avg_color: np.ndarray containing the average color in RGB
    '''
    assert colored_area is not None, "No colored_area given"
    assert len(np.array(colored_area).shape) == 3, f"colored_area is not properly shaped: Expected [x, y, 3], received [x, y, {len(colored_area.shape)}]"

    # Get layers of image
    red_layer = colored_area[:,:, 2]
    green_layer = colored_area[:,:, 1]
    blue_layer = colored_area[:,:, 0] 

    # get average intensity of each layer: 
    bgr_intenisty = np.array([np.mean(blue_layer), np.mean(green_layer), np.mean(red_layer)])
    highest_intenisty_mask = np.array([bgr_intenisty >= np.max(bgr_intenisty)])
    highest_intenisty_layer = int(np.flatnonzero(highest_intenisty_mask))

    # gets pixel_mask for image for highest intensity 
    threshold = cfg.color_filter_tolerance * np.max(colored_area[:,:, highest_intenisty_layer])
    colored_area_layer = colored_area[:, :, highest_intenisty_layer]
    colored_area_mask = np.array(colored_area_layer >= threshold)


    # Apply mask
    filtered = np.zeros_like(colored_area)
    filtered[colored_area_mask] = colored_area[colored_area_mask]

  # cv.imshow("filtered_area", filtered)
  # cv.waitKey(0)
  # cv.destroyAllWindows()

    # Compute average color of the masked pixels
    avg_color = colored_area[colored_area_mask].reshape(-1, 3).mean(axis=0)
    avg_color = avg_color[-1::-1].astype(int)

    return avg_color

def edge_Laplace(image: np.array ) -> np.ndarray:
    '''
    Edge detection applying the Laplace convolution
    Args: 
        image: image to apply Laplace edge detection on.
    Returns: 
        edges_image: image containing the edges.
    '''
    # Laplacian matrix
    L = 0.25 * np.array([[0, -1, 0],[-1, 4, -1], [0, -1, 0]])
    print(f"shape L: {L.shape}")
    print(f"image shape: {image.shape}")
    edges_image =  convolve2d(image, L, mode="same", boundary="symm")

    return edges_image

def crop_image(image: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    ''' 
    Crops the image automatically,  based on the average color taken in the center of the image.
    It assumes that the image is accurately centered.
    Args: 
        - image: image to be cropped
        - filter sensitivity: how sensitive it is going to be to color changes when delimiting the area to be cropped.
        - morph_kernel_dims: tuple of length 4 first 2 elements contains the dimensions of the structural element for the morphological opening of the image. ;ast 2 elements for the morph. close
    Returns: 
        - cropped_image: cropped image
    '''
    assert image is not None, "No image given"
    assert cfg.crop_filter_tolerance >= 0 and cfg.crop_filter_tolerance <= 1, "filter_tolerance has to be in range  [0, 1]"

    # get the color to filter by, returns in RGB
    filter_color = get_avg_color(get_centered_patch(image, cfg)["patch"], cfg)

    # define range of color based on filter tolerance
    red_min = filter_color[0] * (1 - cfg.crop_filter_tolerance);    red_max = filter_color[0] * (1 + cfg.crop_filter_tolerance)
    green_min = filter_color[1] *(1 - cfg.crop_filter_tolerance);  green_max = filter_color[1] * (1 + cfg.crop_filter_tolerance)
    blue_min = filter_color[2] * (1 - cfg.crop_filter_tolerance);   blue_max = filter_color[2] * (1 + cfg.crop_filter_tolerance)

    # define color layers
    red_layer = image[:, :, 2]
    green_layer = image[:, :, 1]
    blue_layer = image[:, :, 0]

    # filter image based on color 
    filter_mask = (
        (red_layer >= red_min) & (red_layer <= red_max) &
        (green_layer >= green_min) & (green_layer <= green_max) &
        (blue_layer >= blue_min) & (blue_layer <= blue_max)
    )

    filtered_image = np.zeros_like(image)
    filtered_image[filter_mask] = image[filter_mask]

    if DEBUGGING:
        cv.imshow("window",filtered_image); cv.waitKey()

    # define morphological structures for opening and closing image
    open_kernel = cv.getStructuringElement(cv.MORPH_RECT, cfg.open_kernel_dims)
    close_kernel = cv.getStructuringElement(cv.MORPH_RECT, cfg.close_kernel_dims)

    # apply open and close to the image
    opened_image = cv.morphologyEx(filtered_image, cv.MORPH_OPEN, open_kernel)
    closed_image = cv.morphologyEx(opened_image, cv.MORPH_CLOSE, close_kernel)

    # binarize image
    gray_image = cv.cvtColor(closed_image, cv.COLOR_BGR2GRAY)
    thresh, binarized = cv.threshold(gray_image, None, 255, cv.THRESH_OTSU)     
    
    # get edges
    edges_image = edge_Laplace(binarized)


    contours, hierarchy = cv.findContours(binarized, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    if DEBUGGING:
        plt.figure()
        for cnt in contours:
            pts = cnt.squeeze()  # contour returns points in [[x, y]] --> [x, y]
            if pts.ndim != 2:
                continue  # skip weird contours
            xs = pts[:, 0]
            ys = pts[:, 1]
            plt.scatter(xs, ys, s=1)  # s small so it doesn’t look like blobs
        plt.gca().invert_yaxis()
        plt.title("Contour points")
        plt.show()

    all_contours = np.concatenate(contours)

    if DEBUGGING:
        print(contours)
        x_bounding, y_bounding, w_bounding, h_bounding = cv.boundingRect(all_contours)
        cv.drawContours(binarized, all_contours, -1, (255, 0, 0), thickness=3 )
        cv.imshow("contours", binarized); cv.waitKey(0)

    cropped_image = None
    return cropped_image
# %% Main ----------------------------------------------------------
def main():

    # Load default config
    cfg = PreprocessConfig()

    # load images of the trays
    trays_directory =(r"C:\Users\Antonio\Documents\GITI\TFG\Trays")
    tray_image = load_images(trays_directory, cfg) 
    print(f"{int(np.shape(tray_image)[0]) + 1} images loaded. \n")

    area_data = get_centered_patch(tray_image[0], cfg); 
    print(area_data)

    # cv.rectangle(tray_image[0], area_data["vortex0"], area_data["vortex1"], (125,0,0), 5)
    # cv.imshow("img0",tray_image[0])
    # cv.waitKey()

    # cv.imshow("2", area_data["patch"])
    # cv.waitKey()

    avg_color = get_avg_color( area_data["patch"], cfg)

    crop_image(tray_image[0], cfg)

    return

if __name__ == "__main__":
    DEBUGGING = True
    main()
    