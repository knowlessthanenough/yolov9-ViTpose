from shapely.geometry import Point, Polygon
import numpy as np

def classify_goalkeeper_behavior(all_frame_detections, distance_threshold=100, movement_threshold=800):
    """
    Classifies goalkeeper behavior for three cases:
    - Class 0 : No any bellow class detected.
    - Class 2: Ball is far from skeleton, and goalkeeper's movement is limited.
    - Class 3: Goalkeeper's last 5-frame average center is farther from the ball than the first 5-frame average center.
    - Class 10: Ball is within the area formed by skeleton points.

    Args:
        all_frame_detections (list): List of frame detections. Each entry is a list of detection dictionaries.
        distance_threshold (float): Maximum average distance for the last 10 frames for classification.
        movement_threshold (float): Maximum movement of the goalkeeper's body center for classification.

    Returns:
        list: A list of integers representing the tags for the video: [1], [2], [3], or combinations.
    """
    body_centers = []  # To track the body center (average of keypoints 5, 6, 11, 12)
    ball_centers = []  # To track the ball's center
    closest_distances = []  # To track closest distances for each frame
    tags = set()  # To store unique tags detected in the video

    for frame_idx, frame_detections in enumerate(all_frame_detections):
        ball_center = None
        goalkeeper_skeleton = None
        tag = 0  # Default tag (no classification)

        # Extract ball center and goalkeeper skeleton for the current frame
        for detection in frame_detections:
            if 'bbox_warp' not in detection or detection['bbox_warp'] is None:
                raise ValueError(f"Detection is missing 'bbox_warp': {detection}")

            if detection['cls'] == 32:  # Ball
                wx1, wy1, wx2, wy2 = detection['bbox_warp']
                ball_center = [(wx1 + wx2) / 2, (wy1 + wy2) / 2]
            if detection['cls'] == 0 and detection['score'] > 0:  # Goalkeeper
                goalkeeper_skeleton = detection['keypoints']

        if ball_center is not None and goalkeeper_skeleton is not None:
            # ------------------------
            # Case 1: Check if the ball is inside the polygon
            # ------------------------
            try:
                polygon_points = [
                    goalkeeper_skeleton[11], goalkeeper_skeleton[12],
                    goalkeeper_skeleton[13], goalkeeper_skeleton[14],
                    goalkeeper_skeleton[15], goalkeeper_skeleton[16]
                ]
                polygon = Polygon(polygon_points)
            except (IndexError, TypeError):
                raise ValueError(f"Incomplete or invalid skeleton keypoints: {goalkeeper_skeleton}")

            ball_point = Point(ball_center)
            if polygon.contains(ball_point):
                tag = 10  # Class 10 detected

            # ------------------------
            # Track closest skeleton point to the ball
            # ------------------------
            distances = [np.linalg.norm(np.array(ball_center) - np.array(point))
                         for point in goalkeeper_skeleton if point is not None]
            if distances:
                closest_distances.append(min(distances))

            # Compute the body center
            try:
                body_points = [
                    goalkeeper_skeleton[5], goalkeeper_skeleton[6],
                    goalkeeper_skeleton[11], goalkeeper_skeleton[12]
                ]
                body_center = np.mean(body_points, axis=0)
                body_centers.append(body_center)
                ball_centers.append(ball_center)
            except (IndexError, TypeError):
                raise ValueError(f"Incomplete or invalid skeleton keypoints for body center: {goalkeeper_skeleton}")

        tags.add(tag)

    # ------------------------
    # Case 2: Check average closest distance and body center movement
    # ------------------------
    if len(closest_distances) >= 10:
        avg_last_10_distance = np.mean(closest_distances[-10:])
        if avg_last_10_distance > distance_threshold:
            # Check overall body movement across all frames
            if len(body_centers) > 1:
                movement_distance = np.linalg.norm(np.array(body_centers[0]) - np.array(body_centers[-1]))
                if movement_distance < movement_threshold:
                    tags.add(2)

    # ------------------------
    # Case 3: Compare average center points
    # ------------------------
    if len(body_centers) >= 10 and len(ball_centers) >= 10:
        avg_body_center_first_5 = np.mean(body_centers[:5], axis=0)
        avg_body_center_last_5 = np.mean(body_centers[-5:], axis=0)
        avg_ball_center_last_5 = np.mean(ball_centers[-5:], axis=0)

        dist_last_to_body = np.linalg.norm(avg_body_center_last_5 - avg_ball_center_last_5)
        dist_first_to_body = np.linalg.norm(avg_body_center_first_5 - avg_ball_center_last_5)

        if dist_last_to_body > dist_first_to_body:
            tags.add(3)

    # Final tag logic
    tags.discard(0)  # Remove 0 if any other tag is present
    return sorted(tags) if tags else [0]  # Return sorted tags or [0] if empty
