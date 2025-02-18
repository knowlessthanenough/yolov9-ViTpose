from shapely.geometry import Point, Polygon
import numpy as np


def classify_goalkeeper_behavior(all_frame_detections, ball_speed, distance_threshold=100, movement_threshold=800, speed_threshold=50, jump_threshold=100, elbow_angle_threshold=160):
    """
    Classifies goalkeeper behavior for three cases:
    - Class 0 : No any bellow class detected.
    - Class 2: Ball is far from skeleton, and goalkeeper's movement is limited.
    - Class 3: Goalkeeper's last 5-frame average center is farther from the ball than the first 5-frame average center.
    - Class 4: Ball above the skeleton nose point but low or not jump to catch the ball.
    - Class 5: Ball above the skeleton nose point but low or not elbow to catch the ball.
    - Class 6: Elbow angle is below the threshold (degrees) when the ball is in the shoulder-centered area.
    - Class 7: Elbow angle is decreasing over time when the ball is in the shoulder-centered area.
    - Class 8: Ball speed is below the threshold (km/h).
    - Class 9: low speed & ball is in the shoulder-centered area.
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
    closest_distances = []  # To track closest skeleton point to the ball for each frame
    nose_positions = []  # To track the goalkeeper's nose positions
    foot_positions = {"left": [], "right": []}  # To track vertical positions of feet (keypoints 15, 16)
    elbow_positions = {"left": [], "right": []}  # To track vertical positions of elbows (keypoints 7, 8)
    tags = set()  # To store unique tags detected in the video
    
    def calculate_angle(a, b, c):
        """
        Calculate the angle (in degrees) between three points a, b, and c.
        - a, b, c: Points in the format [x, y].
        - b is the vertex point.
        """
        ab = np.array(a) - np.array(b)
        cb = np.array(c) - np.array(b)
        dot_product = np.dot(ab, cb)
        norm_ab = np.linalg.norm(ab)
        norm_cb = np.linalg.norm(cb)
        cos_angle = dot_product / (norm_ab * norm_cb + 1e-6)  # Avoid division by zero
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))  # Clip to avoid floating point errors
        return np.degrees(angle)


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
            # Case 10: Check if the ball is inside the polygon
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
            # Case 6 and Case 7 and Case 9: Ball in shoulder-centered area
            # ------------------------
            if goalkeeper_skeleton[5] is not None and goalkeeper_skeleton[6] is not None:
                # Calculate upper arm lengths
                left_upper_arm_length = np.linalg.norm(np.array(goalkeeper_skeleton[5]) - np.array(goalkeeper_skeleton[7]))
                right_upper_arm_length = np.linalg.norm(np.array(goalkeeper_skeleton[6]) - np.array(goalkeeper_skeleton[8]))
                max_radius = 2 * max(left_upper_arm_length, right_upper_arm_length)

                # Check ball in area
                left_shoulder_center = goalkeeper_skeleton[5]
                right_shoulder_center = goalkeeper_skeleton[6]
                distance_to_left = np.linalg.norm(np.array(ball_center) - np.array(left_shoulder_center))
                distance_to_right = np.linalg.norm(np.array(ball_center) - np.array(right_shoulder_center))

                ball_in_area = distance_to_left <= max_radius or distance_to_right <= max_radius

                if ball_in_area:
                    if ball_speed < speed_threshold:
                        tags.add(9)
                        
                    # Calculate elbow angles
                    left_elbow_angle = calculate_angle(goalkeeper_skeleton[5], goalkeeper_skeleton[7], goalkeeper_skeleton[9])
                    right_elbow_angle = calculate_angle(goalkeeper_skeleton[6], goalkeeper_skeleton[8], goalkeeper_skeleton[10])

                    # Track angles over time
                    if frame_idx > 0:
                        if len(elbow_positions["left"]) > 0 and len(elbow_positions["right"]) > 0:
                            prev_left_angle = elbow_positions["left"][-1]
                            prev_right_angle = elbow_positions["right"][-1]

                            # Check if the elbow angle has decreased
                            if left_elbow_angle < prev_left_angle or right_elbow_angle < prev_right_angle:
                                tags.add(7)

                    # Append current angles for next frame comparison
                    elbow_positions["left"].append(left_elbow_angle)
                    elbow_positions["right"].append(right_elbow_angle)

                    # Class 6: Elbow angle < threshold
                    if left_elbow_angle < elbow_angle_threshold or right_elbow_angle < elbow_angle_threshold:
                        tags.add(6)


            # ------------------------
            # Track positions for Classes 4 and 5
            # ------------------------
            if goalkeeper_skeleton[15] is not None:  # Left foot
                foot_positions["left"].append(goalkeeper_skeleton[15][1])  # Track y-coordinates (vertical)
            if goalkeeper_skeleton[16] is not None:  # Right foot
                foot_positions["right"].append(goalkeeper_skeleton[16][1])  # Track y-coordinates (vertical)
            if goalkeeper_skeleton[0] is not None:  # Nose
                nose_positions.append(goalkeeper_skeleton[0][1])  # Track nose y-coordinates
            if goalkeeper_skeleton[7] is not None:  # Left elbow
                elbow_positions["left"].append(goalkeeper_skeleton[7][1])  # Track y-coordinates (vertical)
            if goalkeeper_skeleton[8] is not None:  # Right elbow
                elbow_positions["right"].append(goalkeeper_skeleton[8][1])  # Track y-coordinates (vertical)

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

    # ------------------------
    # Case 4: Check vertical jump of feet and ball height
    # ------------------------
    if foot_positions["left"] and foot_positions["right"] and nose_positions:
        left_foot_vertical_movement = max(foot_positions["left"]) - min(foot_positions["left"])
        right_foot_vertical_movement = max(foot_positions["right"]) - min(foot_positions["right"])
        avg_nose_height = np.mean(nose_positions)
        if (left_foot_vertical_movement + right_foot_vertical_movement) < jump_threshold and ball_center[1] < avg_nose_height:
            tags.add(4)

    # ------------------------
    # Case 5: Check ball height and elbow positions
    # ------------------------
    if elbow_positions["left"] and elbow_positions["right"] and nose_positions:
        max_left_elbow_height = max(elbow_positions["left"])
        max_right_elbow_height = max(elbow_positions["right"])
        avg_nose_height = np.mean(nose_positions)
        if ball_center[1] < avg_nose_height and (max_left_elbow_height < avg_nose_height or max_right_elbow_height < avg_nose_height):
            tags.add(5) 

    # ------------------------
    # Case 8: Compare ball speed
    # ------------------------
    if ball_speed < speed_threshold:
        tags.add(8)

    # Final tag logic
    tags.discard(0)  # Remove 0 if any other tag is present
    return sorted(tags) if tags else [0]  # Return sorted tags or [0] if empty
