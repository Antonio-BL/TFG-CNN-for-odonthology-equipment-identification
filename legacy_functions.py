import numpy as np
def get_avg_color_by_highest_px_intensity(colored_area: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    '''
    Gets the average color of an image. Using a custom method.
    Args: 
        colored_area: image to get the average color of in RGB.
        cfg: Configuration parameters
    Returns: 
        avg_color_rgb: np.ndarray containing the average color in RGB
    '''
    assert colored_area is not None, "No colored_area given"
    assert len(np.array(colored_area).shape) == 3, f"colored_area is not properly shaped: Expected [x, y, 3], received [x, y, {len(colored_area.shape)}]"

    # Get layers of image
    red_layer = colored_area[:,:, 0]
    green_layer = colored_area[:,:, 1]
    blue_layer = colored_area[:,:, 2] 

    # get average intensity of each RGB layer: 
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

    # Compute average color of the masked pixels
    avg_color = colored_area[colored_area_mask].reshape(-1, 3).mean(axis=0)
    avg_color = np.clip(avg_color, 0, 255).astype(np.uint8) 

    return avg_color
