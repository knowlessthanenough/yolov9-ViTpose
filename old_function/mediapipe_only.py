import cv2
import mediapipe as mp

def draw_skeleton_on_image(image_path, output_path=None):
    """
    Detects and draws a skeleton on an image using MediaPipe and OpenCV.

    Args:
        image_path (str): Path to the input image.
        output_path (str, optional): Path to save the output image. If None, the result is displayed.
    """
    # Initialize MediaPipe Pose and Drawing utilities
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at {image_path}")

    # Convert BGR image to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image using MediaPipe Pose
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(image_rgb)

        # Check if landmarks are detected
        if results.pose_landmarks:
            # Draw pose landmarks on the image
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing_styles.get_default_pose_landmarks_style()
            )

    # Save or display the result
    if output_path:
        cv2.imwrite(output_path, image)
        print(f"Skeleton image saved to {output_path}")
    else:
        cv2.imshow("Skeleton", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Test the function
if __name__ == "__main__":
    draw_skeleton_on_image("./runs/detect/yolov9_c_c_640_detect4/debug_crops/goalkeeper_2_crop_364_418.jpg", "./runs/detect/yolov9_c_c_640_detect4/debug_crops/person_skeleton.jpg")