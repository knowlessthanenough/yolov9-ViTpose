import cv2
import numpy as np

def crop_image_with_overlap(image, crop_size=640, overlap=0.2):
    """
    Split a large image into overlapping patches.

    Args:
        image (ndarray): Original image (e.g., 4K image)
        crop_size (int): Patch size (default 640)
        overlap (float): Overlap ratio between 0 and <1 (e.g., 0.2 = 20%)

    Returns:
        List of (patch, (x_offset, y_offset))
    """
    h, w = image.shape[:2]
    stride = int(crop_size * (1 - overlap))
    patches = []

    for y in range(0, h - crop_size + 1, stride):
        for x in range(0, w - crop_size + 1, stride):
            patch = image[y:y + crop_size, x:x + crop_size]
            patches.append((patch, (x, y)))

    # Edge padding if needed (right/bottom borders)
    if (w - crop_size) % stride != 0:
        x = w - crop_size
        for y in range(0, h - crop_size + 1, stride):
            patch = image[y:y + crop_size, x:x + crop_size]
            patches.append((patch, (x, y)))

    if (h - crop_size) % stride != 0:
        y = h - crop_size
        for x in range(0, w - crop_size + 1, stride):
            patch = image[y:y + crop_size, x:x + crop_size]
            patches.append((patch, (x, y)))

    # Bottom-right corner
    if (w - crop_size) % stride != 0 and (h - crop_size) % stride != 0:
        x = w - crop_size
        y = h - crop_size
        patch = image[y:y + crop_size, x:x + crop_size]
        patches.append((patch, (x, y)))

    return patches

if __name__ == "__main__":
    # Example usage
    # Load a 4K image (replace with your image path)
    image = cv2.imread("figure/4k_image.jpg")
    patches = crop_image_with_overlap(image, crop_size=640, overlap=0.2)

    for i, (patch, (x, y)) in enumerate(patches):
        cv2.imwrite(f"images/patch_{i}_at_{x}_{y}.jpg", patch)