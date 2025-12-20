

def show_images(images, titles=None, rows=1, cols=None):
    """
    Display multiple images in a grid using Matplotlib.
    """
    import math
    cols = cols or math.ceil(len(images) / rows)
    plt.figure(figsize=(4 * cols, 4 * rows))
    for i, img in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        if img.ndim == 3:  # color image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)
        else:
            plt.imshow(img, cmap="gray")
        if titles:
            plt.title(titles[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()
    return