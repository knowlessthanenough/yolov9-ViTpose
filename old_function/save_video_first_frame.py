import cv2

def save_first_frame(video_path, output_image_path):
    """
    Save the first frame of a video to an image file.

    Args:
        video_path (str): Path to the input video file.
        output_image_path (str): Path to save the first frame image.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video was successfully opened
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Read the first frame
    ret, frame = cap.read()

    if ret:
        # Save the frame as an image
        cv2.imwrite(output_image_path, frame)
        print(f"First frame saved to {output_image_path}")
    else:
        print("Error: Could not read the first frame of the video.")

    # Release the video capture object
    cap.release()

if __name__ == "__main__":
    video_path = "../data/video/0312behavior7.mp4"
    output_image_path = "../data/images/0312behavior7.jpg"
    save_first_frame(video_path, output_image_path)