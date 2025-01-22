# TODO: ADD score base one skeleton size
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def extract_color_histogram_with_specific_background_color(image, mask=None, bins=(50, 60, 60), ignore_color_range=None):
    """
    Extract the color histogram from an image in HSV format, with the option to ignore specific color ranges.

    Args:
        image (np.ndarray): Input image (BGR format).
        mask (np.ndarray): Optional mask to focus on specific regions.
        bins (tuple): Number of bins for Hue, Saturation, and Value.
        ignore_color_range (tuple): Range of HSV values to ignore ((low_H, low_S, low_V), (high_H, high_S, high_V)).

    Returns:
        np.ndarray: Normalized color histogram.
    """
    # Convert image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # If ignore_color_range is provided, mask out the specified range
    if ignore_color_range:
        lower_bound = np.array(ignore_color_range[0], dtype=np.uint8)
        upper_bound = np.array(ignore_color_range[1], dtype=np.uint8)
        ignore_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

        # Invert the ignore mask to keep only valid colors
        if mask is not None:
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(ignore_mask))
        else:
            mask = cv2.bitwise_not(ignore_mask)

    # Calculate the histogram
    hist = cv2.calcHist([hsv_image], [0, 1, 2], mask, bins, [0, 180, 0, 256, 0, 256])

    # Normalize the histogram
    hist = hist / np.sum(hist)  # Normalize to make it independent of image size
    return hist


def extract_color_histogram_from_rotated_skelton(image, skeleton_keypoints, bins=(50, 60, 60), expand_ratio=0.2, ignore_skin=True):
    """
    Extract the clothes histogram from an image using a rotated bounding box based on skeleton keypoints.

    Args:
        image (np.ndarray): Input image (BGR format).
        skeleton_keypoints (dict): Detected skeleton keypoints with body part names as keys and (x, y) tuples as values.
        bins (tuple): Number of bins for Hue, Saturation, and Value.
        expand_ratio (float): Ratio to expand the bounding box dimensions.
        ignore_skin (bool): Whether to exclude skin tones from the histogram.

    Returns:
        np.ndarray: Normalized clothes histogram.
    """
    # Define torso keypoints
    left_shoulder = skeleton_keypoints["left_shoulder"]
    right_shoulder = skeleton_keypoints["right_shoulder"]
    left_hip = skeleton_keypoints["left_hip"]
    right_hip = skeleton_keypoints["right_hip"]

    # Calculate the center of the torso
    center_x = (left_shoulder[0] + right_shoulder[0] + left_hip[0] + right_hip[0]) / 4
    center_y = (left_shoulder[1] + right_shoulder[1] + left_hip[1] + right_hip[1]) / 4

    # Calculate the torso orientation (angle)
    dx = right_shoulder[0] - left_shoulder[0]
    dy = right_shoulder[1] - left_shoulder[1]
    angle = np.degrees(np.arctan2(dy, dx))

    # Calculate the width and height of the torso bounding box
    torso_width = np.sqrt(dx**2 + dy**2) * (1 + expand_ratio)
    torso_height = np.sqrt((right_hip[0] - left_shoulder[0])**2 + (right_hip[1] - left_shoulder[1])**2) * (1 + expand_ratio)

    # Create a rotated rectangle
    rect = ((center_x, center_y), (torso_width, torso_height), angle)

    # Generate a mask for the rotated rectangle
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    box = cv2.boxPoints(rect).astype(int)  # Convert to corner points
    cv2.fillPoly(mask, [box], 255)

    # Convert to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Exclude skin tones if needed
    if ignore_skin:
        skin_lower = np.array([0, 40, 60], dtype=np.uint8)   # Adjust based on dataset
        skin_upper = np.array([20, 150, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv_image, skin_lower, skin_upper)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(skin_mask))

    # Extract the histogram for the masked region
    hist = cv2.calcHist([hsv_image], [0, 1, 2], mask, bins, [0, 180, 0, 256, 0, 256])

    # Normalize the histogram
    hist = hist / np.sum(hist)
    return hist


def compare_histograms(hist1, hist2, method=cv2.HISTCMP_BHATTACHARYYA):
    """
    Compare two histograms and compute their similarity.

    Args:
        hist1 (np.ndarray): First histogram.
        hist2 (np.ndarray): Second histogram.
        method (int): OpenCV method for histogram comparison (e.g., cv2.HISTCMP_BHATTACHARYYA).
        threshold (float): Minimum similarity score to accept as a match (0-1 for Bhattacharyya).

    Returns:
        float: Similarity score (higher is better).
    """
    # Compare the histograms
    similarity = 1 - cv2.compareHist(hist1, hist2, method)  # Invert for similarity (1 = identical)

    return similarity


def plot_hsv_histogram(hist, bins=(50, 60, 60)):
    """
    Plot the HSV histogram as a 3D bar chart.

    Args:
        hist (np.ndarray): The HSV histogram (normalized).
        bins (tuple): Number of bins for Hue, Saturation, and Value.
    """
    # Extract bin counts
    h_bins, s_bins, v_bins = bins
    h_range = np.linspace(0, 180, h_bins)
    s_range = np.linspace(0, 256, s_bins)
    v_range = np.linspace(0, 256, v_bins)

    # Reshape the histogram for plotting
    hist = hist.reshape((h_bins, s_bins, v_bins))

    # Create a 3D bar chart
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Prepare bar positions
    xpos, ypos, zpos = np.meshgrid(
        np.arange(h_bins),
        np.arange(s_bins),
        np.arange(v_bins),
        indexing="ij"
    )
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos)
    dx = dy = 0.8  # Bar width
    dz = hist.flatten()

    # Normalize bar colors based on Hue and Saturation
    colors = plt.cm.hsv(xpos / h_bins)

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, shade=True)
    ax.set_xlabel('Hue')
    ax.set_ylabel('Saturation')
    ax.set_zlabel('Value')

    plt.title('HSV Histogram')
    plt.show()


def save_histogram(hist, file_path):
    """
    Save the histogram to a file for later use.

    Args:
        hist (np.ndarray): The histogram to save.
        file_path (str): Path to save the histogram file.
    """
    np.save(file_path, hist)
    print(f"Histogram saved to {file_path}")


def load_histogram(file_path):
    """
    Load the histogram from a file.

    Args:
        file_path (str): Path to the saved histogram file.

    Returns:
        np.ndarray: Loaded histogram.
    """
    hist = np.load(file_path)
    print(f"Histogram loaded from {file_path}")
    return hist


if __name__ == "__main__":
    # Define the HSV range for white
    white_range = ((0, 0, 200), (179, 30, 255))  # Low Saturation, High Value

    # Skeleton keypoints for one person (example format)
    skeleton_keypoints = {
    "left_shoulder": (58, 44),
    "right_shoulder": (169, 42),
    "left_hip": (72, 214),
    "right_hip": (152, 213)
    }

    # Load images
    goalkeeper_clothe_image = cv2.imread("./data/images/goalkeeper_clothes.jpeg")

    # Extract histograms, ignoring white
    goalkeeper_hist_from_image = extract_color_histogram_with_specific_background_color(goalkeeper_clothe_image, ignore_color_range=white_range)

    # Extract histograms from rotated bounding box
    goalkeeper_hist_from_skelton = extract_color_histogram_from_rotated_skelton(goalkeeper_clothe_image, skeleton_keypoints)

    score = compare_histograms(goalkeeper_hist_from_image, goalkeeper_hist_from_skelton)  # Should return 1.0

    print(f"Similarity score: {score}")

    # # Plot the histogram
    # plot_hsv_histogram(goalkeeper_hist_from_skelton, bins=(50, 60, 60))


# if __name__ == "__main__":
    # skeleton_keypoints = {
    #     "left_shoulder": (899 , 400),
    #     "right_shoulder": (1000 ,405),
    #     "left_hip": (910 ,525),
    #     "right_hip": (992, 527)
    # }

    # goal_keeper_image = cv2.imread("./data/images/10-1_first_frame.jpg")

    # # #crop the image base on the skeleton keypoints
    # # left_shoulder = skeleton_keypoints["left_shoulder"]
    # # right_shoulder = skeleton_keypoints["right_shoulder"]
    # # left_hip = skeleton_keypoints["left_hip"]
    # # right_hip = skeleton_keypoints["right_hip"]

    # # x1 = int(min(left_shoulder[0], right_shoulder[0], left_hip[0], right_hip[0]))
    # # x2 = int(max(left_shoulder[0], right_shoulder[0], left_hip[0], right_hip[0]))
    # # y1 = int(min(left_shoulder[1], right_shoulder[1], left_hip[1], right_hip[1]))
    # # y2 = int(max(left_shoulder[1], right_shoulder[1], left_hip[1], right_hip[1]))

    # # goal_keeper_image = goal_keeper_image[y1:y2, x1:x2]

    # # cv2.imshow("goalkeeper", goal_keeper_image)
    # # cv2.waitKey(0)

    # # goalkeeper_hist_from_skelton = extract_color_histogram_from_rotated_skelton(goal_keeper_image, skeleton_keypoints)

    # # save_histogram(goalkeeper_hist_from_skelton, "./data/histograms/10-1_goalkeeper_hist.npy")

    # # plot_hsv_histogram(goalkeeper_hist_from_skelton, bins=(50, 60, 60))

