import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import defaultdict
from scipy.signal import savgol_filter
import cv2
import os
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import time
from homography_matrix import compute_homography
from filterpy.kalman import KalmanFilter
from typing import List, Dict, Any, Tuple


def interpolate_missing_frames(frames, points):
    """
    Fill in missing frames using linear interpolation.

    Args:
        frames (list[int]): Existing frame indices.
        points (list[list[float]]): Corresponding (x, y) positions.

    Returns:
        tuple: (interpolated_frames, interpolated_points)
    """
    if len(frames) != len(points):
        raise ValueError("frames and points must be the same length")
    
    frame_arr = np.array(frames)
    point_arr = np.array(points)

    full_frames = np.arange(frame_arr[0], frame_arr[-1] + 1)
    x_interp = np.interp(full_frames, frame_arr, point_arr[:, 0])
    y_interp = np.interp(full_frames, frame_arr, point_arr[:, 1])
    
    interpolated_points = np.stack([x_interp, y_interp], axis=1)

    return full_frames.tolist(), interpolated_points.tolist()


def smooth_and_project_bboxes_with_ekf(bboxes, H):
    """
    Smooth bbox height using EKF and project the bottom-center points using the homography matrix H.

    Args:
        bboxes (List[List[float]]): A list of bounding boxes in [x1, y1, x2, y2] format.
        H (np.ndarray): Homography matrix for coordinate projection.

    Returns:
        projected_points (np.ndarray): Array of projected bottom-center points (Nx2).
    """
    if len(bboxes) == 0:
        return np.empty((0, 2))

    bboxes = np.array(bboxes)
    heights = bboxes[:, 3] - bboxes[:, 1]  # y2 - y1
    centers_x = (bboxes[:, 0] + bboxes[:, 2]) / 2
    bottoms_y = bboxes[:, 3]

    # Initialize Kalman Filter for height smoothing
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([heights[0], 0])          # Initial state (height, velocity)
    kf.F = np.array([[1, 1], [0, 1]])          # State transition matrix
    kf.H = np.array([[1, 0]])                  # Measurement function
    kf.P *= 10.                              # Initial uncertainty
    kf.R *= 50                                  # Measurement noise
    kf.Q = np.array([[1, 0], [0, 1]]) * 0.0001    # Process noise

    smoothed_heights = []
    for h in heights:
        kf.predict()
        kf.update(h)
        smoothed_heights.append(kf.x[0])

    smoothed_heights = np.array(smoothed_heights)
    bottom_points = np.stack([centers_x, bottoms_y], axis=1)

    # Adjust y = y1 + smoothed height
    bottom_points[:, 1] = bboxes[:, 1] + smoothed_heights

    # Project using homography
    ones = np.ones((bottom_points.shape[0], 1))
    points_homo = np.hstack([bottom_points, ones])
    projected_homo = (H @ points_homo.T).T
    projected = projected_homo[:, :2] / projected_homo[:, 2:]

    return projected


def apply_position_smoothing(xs, ys, method='savgol', window_size=7, polyorder=2, max_step=20):
    """
    修補跳變點並平滑座標軌跡。

    Args:
        xs, ys (np.ndarray): 原始座標序列。
        method (str): 平滑方法，目前只支援 'savgol'。
        window_size (int): 平滑窗口大小。
        polyorder (int): Savitzky-Golay 多項式階數。
        max_step (float): 相鄰點距離超過此值會被視為 outlier 並修補。

    Returns:
        xs_smooth, ys_smooth: 修補並平滑後的座標序列。
    """
    xs, ys = np.array(xs), np.array(ys)
    new_xs, new_ys = xs.copy(), ys.copy()

    for i in range(1, len(xs)):
        dx = xs[i] - xs[i - 1]
        dy = ys[i] - ys[i - 1]
        dist = np.hypot(dx, dy)

        if dist > max_step:
            # 修補跳點，用前後點平均（若可），否則用上一點
            if 1 < i < len(xs) - 1:
                new_xs[i] = (xs[i - 1] + xs[i + 1]) / 2
                new_ys[i] = (ys[i - 1] + ys[i + 1]) / 2
            else:
                new_xs[i] = new_xs[i - 1]
                new_ys[i] = new_ys[i - 1]

    if method == 'savgol' and len(xs) >= window_size:
        new_xs = savgol_filter(new_xs, window_length=window_size, polyorder=polyorder)
        new_ys = savgol_filter(new_ys, window_length=window_size, polyorder=polyorder)

    return new_xs, new_ys


def assign_team_by_majority_vote(team_conf_list):
    team_count = defaultdict(float)
    for conf in team_conf_list:
        for k, v in conf.items():
            team_count[k] += v
    return max(team_count, key=team_count.get) if team_count else "ball"


def index_to_letter_suffix(idx):
    """Return 'a', 'b', ..., 'z', 'aa', 'ab', ... as suffix."""
    letters = []
    while True:
        letters.append(chr(97 + (idx % 26)))
        idx = idx // 26
        if idx == 0:
            break
        idx -= 1  # offset for 0-based index
    return ''.join(reversed(letters))


def create_segment(track_id_base, segment_index, frame_ids, projected, bboxes, team_conf):
    """
    Creates a track segment dictionary from the given inputs.
    """
    return {
        "track_id": f"{track_id_base}{index_to_letter_suffix(segment_index)}",
        "frame_id": frame_ids,
        "projected": projected,
        "bbox": bboxes if bboxes else [],
        "team_conf": team_conf,
        "team": assign_team_by_majority_vote(team_conf)
    }


def split_track_by_team_conf(obj, min_segment_length=5):
    """
    Splits a single track into segments based on team_conf changes,
    merging short segments into the previous one.

    Args:
        obj (dict): Track data.
        threshold_ratio (float): Not used in this version but kept for API compatibility.
        min_segment_length (int): Minimum frames to consider a segment valid.

    Returns:
        List[dict]: List of split (or merged) track segments.
    """
    team_conf_list = obj["team_conf"]
    frame_ids = obj["frame_id"]
    projected = obj["projected"]
    bboxes = obj.get("bbox", [])

    segments = []
    current_team = None
    buffer = []

    for i, conf in enumerate(team_conf_list):
        dominant = max(conf, key=conf.get) if conf else "ball"

        if dominant != current_team:
            if buffer:
                if len(buffer) < min_segment_length and segments:
                    # Merge short segment into previous
                    prev = segments[-1]
                    prev["frame_id"].extend([frame_ids[j] for j in buffer])
                    prev["projected"].extend([projected[j] for j in buffer])
                    if bboxes:
                        prev["bbox"].extend([bboxes[j] for j in buffer])
                    prev["team_conf"].extend([team_conf_list[j] for j in buffer])
                    prev["team"] = assign_team_by_majority_vote(prev["team_conf"])
                else:
                    segment = create_segment(
                        obj["track_id"],
                        len(segments),
                        [frame_ids[j] for j in buffer],
                        [projected[j] for j in buffer],
                        [bboxes[j] for j in buffer] if bboxes else [],
                        [team_conf_list[j] for j in buffer]
                    )
                    segments.append(segment)
            buffer = [i]
            current_team = dominant
        else:
            buffer.append(i)

    # Add last buffer
    if buffer:
        if len(buffer) < min_segment_length and segments:
            prev = segments[-1]
            prev["frame_id"].extend([frame_ids[j] for j in buffer])
            prev["projected"].extend([projected[j] for j in buffer])
            if bboxes:
                prev["bbox"].extend([bboxes[j] for j in buffer])
            prev["team_conf"].extend([team_conf_list[j] for j in buffer])
            prev["team"] = assign_team_by_majority_vote(prev["team_conf"])
        else:
            segment = create_segment(
                obj["track_id"],
                len(segments),
                [frame_ids[j] for j in buffer],
                [projected[j] for j in buffer],
                [bboxes[j] for j in buffer] if bboxes else [],
                [team_conf_list[j] for j in buffer]
            )
            segments.append(segment)

    return segments


def interpolate_full_track(frames: List[int], points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate full track to fill in all missing frames using linear interpolation.

    Args:
        frames (List[int]): List of frame indices.
        points (np.ndarray): Corresponding points (N, 2) for each frame.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Interpolated frames and points (in same order).
    """
    if len(frames) < 2:
        return np.array(frames), points

    all_frames = np.arange(frames[0], frames[-1] + 1)
    xs_interp = np.interp(all_frames, frames, points[:, 0])
    ys_interp = np.interp(all_frames, frames, points[:, 1])
    full_points = np.stack([xs_interp, ys_interp], axis=1)

    return all_frames, full_points


def merge_tracks_with_recursion(track_dict: Dict[str, Dict[str, Any]],
                                max_merge_gap: int = 5,
                                max_merge_overlap_frames: int = 3,
                                max_merge_distance: float = 10) -> List[Dict[str, Any]]:
    """
    Recursively merges track segments based on temporal and spatial proximity with optional interpolation.

    Args:
        track_dict (dict): Dictionary where key is track_id and value is a dict containing 'team', 'frames', and 'points'.
        max_merge_gap (int): Maximum frame gap to consider for merging.
        max_merge_distance (float): Maximum distance to consider for merging.

    Returns:
        list: List of merged track segments.
    """
    track_items = sorted(track_dict.items(), key=lambda x: x[1]["frames"][0])
    merged_tracks = []
    used = set()

    for i, (tid_a, data_a) in enumerate(track_items):
        if tid_a in used:
            continue
        merged = {
            "team": data_a["team"],
            "frames": list(data_a["frames"]),
            "points": list(data_a["points"]),
            "track_id": tid_a
        }
        used.add(tid_a)

        while True:
            best_candidate = None
            best_dist = float('inf')
            for j, (tid_b, data_b) in enumerate(track_items):
                if tid_b in used or data_b["team"] != merged["team"]:
                    continue
                # Check if the gap is within the allowed range for merging
                '''
                If overlap < 0:
                data_b starts before base_track ends → overlap
                Allow if overlap <= max_overlap_frames

                If overlap > 0:
                data_b starts after base_track ends → gap
                Allow if abs(overlap) <= max_merge_gap
                '''
                gap = data_b["frames"][0] - merged["frames"][-1]
                if (0 <= gap <= max_merge_gap) or (0 < -gap <= max_merge_overlap_frames):
                    dist = np.linalg.norm(np.array(merged["points"][-1]) - np.array(data_b["points"][0]))
                    if dist <= max_merge_distance and dist < best_dist:
                        best_candidate = (tid_b, data_b)
                        best_dist = dist
            if best_candidate is None:
                break

            tid_b, data_b = best_candidate
            merged["frames"].extend(data_b["frames"])
            merged["points"].extend(data_b["points"])
            used.add(tid_b)

        merged_tracks.append(merged)
        for track in merged_tracks:
            frames, points = interpolate_full_track(track["frames"], np.array(track["points"]))
            track["frames"] = frames
            track["points"] = points

    return merged_tracks


def load_and_merge_tracks(
    json_path,
    field_size,
    min_track_length,
    smoothing_window,
    polyorder,
    max_step,
    max_merge_gap,
    max_merge_overlap_frames,
    max_merge_distance,
    min_segment_length
):
    """
    Draw smoothed 2D trajectories from tracking JSON with optional merging of fragmented tracks.

    Args:
        json_path (str): Path to tracking JSON.
        image_path (str): Path to field background image.
        field_size (tuple): Field dimension (width, height).
        min_track_length (int): Minimum track length to visualize.
        smoothing_window (int): Window size for Savitzky-Golay smoothing.
        polyorder (int): Polynomial order for smoothing.
        max_merge_gap (int): Max frame gap between track ends to consider merging.
        max_merge_distance (float): Max distance in projected space to consider merging.


    Returns:
        merged_tracks (list): List of merged track dicts with keys 'track_id', 'team', 'frames', and 'points'.
        bg_img (np.ndarray): The resized field background image.
    """
    # Load JSON data
    with open(json_path) as f:
        data = json.load(f)

    # Collect and smooth trajectories
    track_dict = {}
    for obj in data:
        split_objects = split_track_by_team_conf(obj, min_segment_length=min_segment_length)

        for split_obj in split_objects:
            tid = split_obj['track_id']
            frames = split_obj['frame_id']
            projected_points = split_obj.get('projected', [])

            if len(projected_points) < min_track_length:
                continue

            pts = np.array([pt for pt in projected_points if pt is not None])
            if len(pts) < min_track_length:
                continue

            xs, ys = pts[:, 0], pts[:, 1]
            in_bounds = (xs >= 0) & (xs <= field_size[0]) & (ys >= 0) & (ys <= field_size[1])
            if in_bounds.sum() < min_track_length:
                continue

            xs, ys = xs[in_bounds], ys[in_bounds]
            frs = np.array(frames)[in_bounds]

            xs, ys = apply_position_smoothing(
                xs, ys,
                method='savgol',
                window_size=smoothing_window,
                polyorder=polyorder,
                max_step=max_step
            )

            track_dict[tid] = {
                "team": split_obj.get("team", "ball"),
                "frames": frs,
                "points": np.stack([xs, ys], axis=1)
            }

    # Merge tracks
    merged_tracks = merge_tracks_with_recursion(track_dict=track_dict, max_merge_gap=max_merge_gap, max_merge_overlap_frames=max_merge_overlap_frames, max_merge_distance=max_merge_distance)

    return merged_tracks


def draw_merged_paths_from_json(
    json_path,
    image_path,
    field_size,
    min_track_length,
    smoothing_window,
    polyorder,
    max_step,
    max_merge_gap,
    max_merge_overlap_frames,
    max_merge_distance,
    min_segment_length
):
    
    # Load and resize field background
    bg_img = cv2.imread(image_path)
    if bg_img is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
    bg_img = cv2.resize(bg_img, field_size)

    merged_tracks= load_and_merge_tracks(
        json_path,
        field_size=field_size,
        min_track_length=min_track_length,
        smoothing_window=smoothing_window,
        polyorder=polyorder,
        max_merge_gap=max_merge_gap,
        max_merge_distance=max_merge_distance,
        min_segment_length=min_segment_length,
        max_step=max_step,
        max_merge_overlap_frames=max_merge_overlap_frames
    )

    # Draw on background
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(bg_img, extent=[0, field_size[0], 0, field_size[1]])
    team_colors = {
        'eastern': 'blue',
        'easterngoalkeeper': 'green',
        'kitchee': 'pink',
        'kitcheegoalkeeper': 'orange',
        'referee': 'yellow',
        'ball': 'black',
    }

    for track in merged_tracks:
        team = track["team"]
        points = np.array(track["points"])
        if len(points) < min_track_length:
            continue
        xs, ys = points[:, 0], points[:, 1]
        color = team_colors.get(team, 'gray')
        ax.plot(xs, ys, color=color, alpha=0.8)
        ax.scatter(xs[-1], ys[-1], color=color)
        ax.text(xs[-1], ys[-1], str(track["track_id"]), fontsize=8, color='black')

    ax.set_xlim(0, field_size[0])
    ax.set_ylim(0, field_size[1])
    ax.set_title("Smoothed & Merged Trajectories")
    plt.tight_layout()
    plt.savefig("trajectory_plot.png", dpi=300)
    plt.show()
    

if  __name__ == "__main__":
    start = time.time()
    # Example usage
    draw_merged_paths_from_json(
        "./runs/detect/test_4k_3-crop-from-video/team_tracking.json", 
        "./data/images/mongkok_football_field.png",
        field_size=(1060, 660),
        min_track_length=10,
        smoothing_window=90,
        polyorder=7,
        max_step=20,
        max_merge_gap=20,
        max_merge_overlap_frames=10,
        max_merge_distance=50,
        min_segment_length=20
        )
    end = time.time()
    print(f"Execution time: {end - start:.2f} seconds")
