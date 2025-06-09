import json
import numpy as np
import cv2
from collections import defaultdict
from scipy.signal import savgol_filter
import time
from load_and_plot_tracks_combine import apply_position_smoothing, assign_team_by_majority_vote, split_track_by_team_conf

def create_current_dot_video(
    json_path,
    image_path,
    output_video_path,
    field_size=(1060, 660),
    min_track_length=5,
    smoothing_window=90,
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

    # Collect and smooth trajectories
    track_dict = {}
    for obj in data:
        split_objects = split_track_by_team_conf(obj, min_segment_length=20)

        for split_obj in split_objects:
            tid = split_obj['track_id']
            frames = split_obj['frame_id']
            projected_points = split_obj.get('projected', [])

            if len(projected_points) < min_track_length:
                continue

            # projected_points = smooth_and_project_bboxes_with_ekf(bbox_sequence, H)

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
                max_step=20
            )

            # print(f"Track {tid} smoothed: {len(xs)} points after smoothing")
            # print(split_obj.get("team"))
            track_dict[tid] = {
                "team": split_obj.get("team", "ball"),
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
