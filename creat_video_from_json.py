import json
import numpy as np
import cv2
from collections import defaultdict
from scipy.signal import savgol_filter
import time
from load_and_plot_tracks_combine import apply_position_smoothing

def create_current_dot_video(
    json_path,
    image_path,
    output_video_path,
    field_size=(1060, 660),
    min_track_length=5,
    smoothing_window=7,
    polyorder=2,
    max_merge_gap=30,
    max_merge_distance=80,
    fps=30
):
    # Load JSON data
    with open(json_path) as f:
        data = json.load(f)

    # Load and resize background
    bg_img = cv2.imread(image_path)
    if bg_img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    bg_img = cv2.resize(bg_img, field_size)
    height, width, _ = bg_img.shape

    # Assign team labels by majority vote
    team_votes = {}
    for obj in data:
        vote = defaultdict(float)
        for conf in obj['team_conf']:
            for k, v in conf.items():
                vote[k] += v
        team_votes[obj['track_id']] = max(vote, key=vote.get) if vote else "ball"

    # Extract smoothed tracks
    track_dict = {}
    for obj in data:
        tid = obj['track_id']
        frames = obj['frame_id']
        proj = obj.get('projected', [])
        pts = np.array([pt for pt in proj if pt is not None])
        if len(pts) < min_track_length:
            continue
        xs, ys = pts[:, 0], pts[:, 1]
        in_bounds = (xs >= 0) & (xs <= field_size[0]) & (ys >= 0) & (ys <= field_size[1])
        if in_bounds.sum() < min_track_length:
            continue
        xs, ys = apply_position_smoothing(
            xs[in_bounds], ys[in_bounds],
            method='savgol',
            window_size=smoothing_window,
            polyorder=polyorder
        )
        frs = np.array(frames)[in_bounds]
        if len(xs) >= smoothing_window:
            xs = savgol_filter(xs, smoothing_window, polyorder)
            ys = savgol_filter(ys, smoothing_window, polyorder)
        track_dict[tid] = {
            "team": team_votes.get(tid, "ball"),
            "frames": frs,
            "points": np.stack([xs, ys], axis=1)
        }

    # Merge tracks with the same team, based on proximity and frame gap
    merged_tracks = []
    used = set()
    track_items = sorted(track_dict.items(), key=lambda x: x[1]["frames"][0])
    for i, (tid_a, a) in enumerate(track_items):
        if tid_a in used:
            continue
        merged = {"track_id": tid_a, "team": a["team"], "frames": list(a["frames"]), "points": list(a["points"])}
        used.add(tid_a)
        for j in range(i + 1, len(track_items)):
            tid_b, b = track_items[j]
            if tid_b in used or a["team"] != b["team"]:
                continue
            if 0 < b["frames"][0] - merged["frames"][-1] <= max_merge_gap:
                dist = np.linalg.norm(np.array(merged["points"][-1]) - np.array(b["points"][0]))
                if dist <= max_merge_distance:
                    merged["frames"].extend(b["frames"])
                    merged["points"].extend(b["points"])
                    used.add(tid_b)
        merged_tracks.append(merged)

    # Set up video writer
    writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    team_colors = {
        'eastern': (255, 0, 0),
        'easterngoalkeeper': (0, 255, 0),
        'kitchee': (255, 192, 203),
        'kitcheegoalkeeper': (0, 165, 255),
        'referee': (0, 255, 255),
        'ball': (0, 0, 0),
    }

    # Define frame range
    min_frame = min(min(t["frames"]) for t in merged_tracks)
    max_frame = max(max(t["frames"]) for t in merged_tracks)

    for f in range(min_frame, max_frame + 1):
        frame_img = bg_img.copy()
        for track in merged_tracks:
            frames = np.array(track["frames"])
            points = np.array(track["points"])
            idx = np.where(frames == f)[0]
            if len(idx) > 0:
                i = idx[0]
                x, y = int(points[i][0]), field_size[1] - int(points[i][1])
                color = team_colors.get(track["team"], (128, 128, 128))
                cv2.circle(frame_img, (x, y), 5, color, -1)
        writer.write(frame_img)

    writer.release()
    print(f"✅ Saved video to: {output_video_path}")

if __name__ == "__main__":
    start = time.time()
    # Example usage
    create_current_dot_video(
        json_path="./runs/detect/test_4k_3-crop-from-video/team_tracking.json",
        image_path="./data/images/mongkok_football_field.png",
        output_video_path="./trajectory_current_only.mp4",
        smoothing_window=90,
        polyorder=7
    )
    end = time.time()
    print(f"Execution time: {end - start:.2f} seconds")
