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

def draw_merged_paths_from_json(
    json_path,
    image_path,
    field_size=(1060, 660),
    min_track_length=5,
    smoothing_window=7,
    polyorder=2,
    max_merge_gap=30,
    max_merge_distance=80
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
    """
    # Load data
    with open(json_path) as f:
        data = json.load(f)

    # Load and resize field background
    bg_img = cv2.imread(image_path)
    if bg_img is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
    bg_img = cv2.resize(bg_img, field_size)

    # Assign team labels by majority vote
    team_votes = {}
    for obj in data:
        team_conf = obj['team_conf']
        team_count = defaultdict(float)
        for conf in team_conf:
            for k, v in conf.items():
                team_count[k] += v
        if team_count:
            assigned_team = max(team_count, key=team_count.get)
            team_votes[obj['track_id']] = assigned_team

    # Collect and smooth trajectories
    track_dict = {}
    for obj in data:
        tid = obj['track_id']
        frames = obj['frame_id']
        proj = obj.get('projected', [])
        if len(proj) < min_track_length:
            continue

        pts = np.array([pt for pt in proj if pt is not None])
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
            max_step=2
        )

        track_dict[tid] = {
            "team": team_votes.get(tid, "ball"),
            "frames": frs,
            "points": np.stack([xs, ys], axis=1)
        }

    # Merge tracks by distance and frame proximity
    merged_tracks = []
    used = set()

    track_items = sorted(track_dict.items(), key=lambda x: x[1]["frames"][0])
    for i, (tid_a, data_a) in enumerate(track_items):
        if tid_a in used:
            continue
        merged = {
            "team": data_a["team"],
            "frames": list(data_a["frames"]),
            "points": list(data_a["points"]),
            "track_id": tid_a  # add this
        }
        used.add(tid_a)

        for j in range(i + 1, len(track_items)):
            tid_b, data_b = track_items[j]
            if tid_b in used or data_a["team"] != data_b["team"]:
                continue
            if 0 < data_b["frames"][0] - merged["frames"][-1] <= max_merge_gap:
                dist = np.linalg.norm(np.array(merged["points"][-1]) - np.array(data_b["points"][0]))
                if dist <= max_merge_distance:
                    merged["frames"].extend(data_b["frames"])
                    merged["points"].extend(data_b["points"])
                    used.add(tid_b)

        merged_tracks.append(merged)

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
    # Example usage
    draw_merged_paths_from_json("./runs/detect/test_4k_3-crop-from-video/team_tracking.json", "./data/images/mongkok_football_field.png", smoothing_window= 80)
