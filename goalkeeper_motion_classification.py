from shapely.geometry import Point, Polygon
import numpy as np

def classify_goalkeeper_behavior(
    all_frame_detections,
    ball_speed,
    distance_threshold,
    movement_threshold,
    speed_threshold,
    jump_threshold,
    elbow_angle_threshold,
):
    """
    Classifies goalkeeper behavior for multiple cases (0,2,...,10).
    If no goalkeeper or no ball is detected at all, returns [0].
    Otherwise, collects final tags (2..10) or [0] if none.

    Classifies goalkeeper behavior for three cases:
    - Class 0 : No any bellow class detected.
    - Class 1 & 2: Ball is far from skeleton, and goalkeeper's movement is limited.
    - Class 3: Goalkeeper's last 5-frame average center is farther from the ball than the first 5-frame average center.
    - Class 4: Ball above the skeleton ear point but low or not jump to catch the ball.
    - Class 5: Ball above the skeleton ear point but low or not raise elbow above ear to catch the ball.
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
    seen_ball = False
    seen_goalkeeper = False

    body_centers = []         # Tracks GK center over frames
    ball_centers = []         # Tracks ball center over frames
    closest_distances = []    # Min dist from ball to GK skeleton
    ear_positions = []       # GK ear y-values
    foot_positions = {"left": [], "right": []}
    elbow_positions_y = {"left": [], "right": []}
    elbow_angles = {"left": [], "right": []}

    tags = set()

    def calculate_angle(a, b, c):
        ab = np.array(a) - np.array(b)
        cb = np.array(c) - np.array(b)
        dot_product = np.dot(ab, cb)
        norm_ab = np.linalg.norm(ab)
        norm_cb = np.linalg.norm(cb)
        denominator = (norm_ab * norm_cb) + 1e-6  # Avoid zero division
        cos_angle = dot_product / denominator
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)

    # ------------------------
    # Process each frame
    # ------------------------
    for frame_idx, frame_detections in enumerate(all_frame_detections):
        ball_center = None
        goalkeeper_skeleton = None

        # Identify ball & GK in this frame
        for detection in frame_detections:
            if 'bbox_warp' not in detection or detection['bbox_warp'] is None:
                continue

            if detection['cls'] == 32:  # Ball
                seen_ball = True
                wx1, wy1, wx2, wy2 = detection['bbox_warp']
                ball_center = [(wx1 + wx2) / 2, (wy1 + wy2) / 2]

            if detection['cls'] == 0 and detection['score'] > 0:  # Goalkeeper
                seen_goalkeeper = True
                goalkeeper_skeleton = detection['keypoints']

        # If we don't have a ball or GK for this frame, skip to the next
        if ball_center is None or goalkeeper_skeleton is None:
            continue

        # -------------- Case 10 --------------
        try:
            polygon_points = [
                goalkeeper_skeleton[11], goalkeeper_skeleton[12],
                goalkeeper_skeleton[13], goalkeeper_skeleton[14],
                goalkeeper_skeleton[15], goalkeeper_skeleton[16]
            ]
            # use remaining points if some are missing
            polygon_points = [pt for pt in polygon_points if pt is not None]
            polygon = Polygon(polygon_points)
            ball_point = Point(ball_center)

            if polygon.is_valid and polygon.contains(ball_point):
                tags.add(10)
        except Exception:
            pass

        # -------------- Cases 6,7,9 --------------
        left_shoulder = goalkeeper_skeleton[5]
        right_shoulder = goalkeeper_skeleton[6]
        if left_shoulder is not None and right_shoulder is not None:
            try:
                left_upper_arm_len = np.linalg.norm(
                    np.array(left_shoulder) - np.array(goalkeeper_skeleton[7])
                )
                right_upper_arm_len = np.linalg.norm(
                    np.array(right_shoulder) - np.array(goalkeeper_skeleton[8])
                )
                max_radius = 2 * max(left_upper_arm_len, right_upper_arm_len)

                dist_left = np.linalg.norm(np.array(ball_center) - np.array(left_shoulder))
                dist_right = np.linalg.norm(np.array(ball_center) - np.array(right_shoulder))

                ball_in_area = dist_left <= max_radius or dist_right <= max_radius

                if ball_in_area:
                    # Case 9: ball speed < threshold
                    if ball_speed < speed_threshold:
                        tags.add(9)

                    # Calculate elbow angles
                    try:
                        left_elbow_angle = calculate_angle(
                            goalkeeper_skeleton[5],
                            goalkeeper_skeleton[7],
                            goalkeeper_skeleton[9]
                        )
                    except (TypeError, ValueError):
                        left_elbow_angle = 180

                    try:
                        right_elbow_angle = calculate_angle(
                            goalkeeper_skeleton[6],
                            goalkeeper_skeleton[8],
                            goalkeeper_skeleton[10]
                        )
                    except (TypeError, ValueError):
                        right_elbow_angle = 180

                    # Case 7: if elbow angle is decreasing over frames
                    if elbow_angles["left"] and elbow_angles["right"]:
                        prev_left = elbow_angles["left"][-1]
                        prev_right = elbow_angles["right"][-1]
                        if left_elbow_angle < prev_left or right_elbow_angle < prev_right:
                            tags.add(7)

                    # Store angles
                    elbow_angles["left"].append(left_elbow_angle)
                    elbow_angles["right"].append(right_elbow_angle)

                    # Case 6: elbow angle < threshold
                    if (left_elbow_angle < elbow_angle_threshold or
                            right_elbow_angle < elbow_angle_threshold):
                        tags.add(6)

            except Exception:
                pass

        # Track foot, ear, and elbow y positions
        if goalkeeper_skeleton[15] is not None:
            foot_positions["left"].append(goalkeeper_skeleton[15][1])
        if goalkeeper_skeleton[16] is not None:
            foot_positions["right"].append(goalkeeper_skeleton[16][1])

        if goalkeeper_skeleton[3] is not None:
            ear_positions.append(goalkeeper_skeleton[3][1])
        if goalkeeper_skeleton[4] is not None:
            ear_positions.append(goalkeeper_skeleton[4][1])

        if goalkeeper_skeleton[7] is not None:
            elbow_positions_y["left"].append(goalkeeper_skeleton[7][1])
        if goalkeeper_skeleton[8] is not None:
            elbow_positions_y["right"].append(goalkeeper_skeleton[8][1])

        # Closest skeleton point to the ball
        distances = [
            np.linalg.norm(np.array(ball_center) - np.array(pt))
            for pt in goalkeeper_skeleton
            if pt is not None
        ]
        if distances:
            closest_distances.append(min(distances))

        # Compute GK body center
        try:
            body_points = [
                goalkeeper_skeleton[5], goalkeeper_skeleton[6],
                goalkeeper_skeleton[11], goalkeeper_skeleton[12]
            ]
            if any(bp is None for bp in body_points):
                raise ValueError("Incomplete body points.")
            body_center = np.mean(body_points, axis=0)
            body_centers.append(body_center)
            ball_centers.append(ball_center)
        except Exception:
            pass

    # ------------------------
    # If we never saw ball or GK, return [0]
    # ------------------------
    if not seen_ball or not seen_goalkeeper:
        return [0]

    # ------------------------
    # Case 2: average distance + limited movement
    # ------------------------
    if len(closest_distances) >= 10:
        avg_last_10 = np.mean(closest_distances[-10:])
        if avg_last_10 > distance_threshold:
            if len(body_centers) > 1:
                total_movement = np.linalg.norm(body_centers[0] - body_centers[-1])
                if total_movement < movement_threshold:
                    tags.add(1)
                    tags.add(2)

    # ------------------------
    # Case 3: Compare average center points
    # ------------------------
    if len(body_centers) >= 10 and len(ball_centers) >= 10:
        first_5_body = np.mean(body_centers[:5], axis=0)
        last_5_body  = np.mean(body_centers[-5:], axis=0)
        last_5_ball  = np.mean(ball_centers[-5:], axis=0)

        dist_last_to_body = np.linalg.norm(last_5_body - last_5_ball)
        dist_first_to_body = np.linalg.norm(first_5_body - last_5_ball)

        if dist_last_to_body > dist_first_to_body:
            tags.add(3)

    # ------------------------
    # Case 4: Ball above ear + minimal foot jump
    # ------------------------
    # Only do this if we have at least one ball center & foot, ear data
    if ball_centers and foot_positions["left"] and foot_positions["right"] and ear_positions:
        last_ball_center = ball_centers[-1]
        left_foot_move = max(foot_positions["left"]) - min(foot_positions["left"])
        right_foot_move = max(foot_positions["right"]) - min(foot_positions["right"])
        total_foot_move = left_foot_move + right_foot_move
        avg_nose_height = np.mean(ear_positions)

        # If your coordinate system is top=0, smaller y means higher
        # so "ball above ear" is ball_center[1] < nose_y
        if total_foot_move < jump_threshold and last_ball_center[1] < avg_nose_height:
            tags.add(4)

    # ------------------------
    # Case 5: Ball above ear + elbows not raised
    # ------------------------
    if ball_centers and elbow_positions_y["left"] and elbow_positions_y["right"] and ear_positions:
        last_ball_center = ball_centers[-1]
        max_left_elbow = max(elbow_positions_y["left"])
        max_right_elbow = max(elbow_positions_y["right"])
        avg_nose_height = np.mean(ear_positions)

        if last_ball_center[1] < avg_nose_height and (
           max_left_elbow < avg_nose_height or max_right_elbow < avg_nose_height):
            tags.add(5)

    # ------------------------
    # Case 8: Low ball speed
    # ------------------------
    if ball_speed < speed_threshold:
        tags.add(8)

    # ------------------------
    # Final
    # ------------------------
    if 0 in tags and len(tags) > 1:
        tags.discard(0)

    return sorted(tags) if tags else [0]
