import json
import numpy as np
import cv2
from collections import defaultdict
from scipy.signal import savgol_filter
import time
from yolov9.output_video_or_imagre_or_json_from_orginal_json import load_and_merge_tracks, remove_referee_near_boundary, save_tracks_to_json
import os


def create_current_dot_video(
    json_path,
    image_path,
    output_video_path,
    field_size,
    min_track_length,
    smoothing_window,
    polyorder,
    max_step,
    max_merge_gap,
    max_merge_overlap_frames,
    max_merge_distance,
    window_size,
    threshold,
    fps=30
):

    # Load and resize background
    bg_img = cv2.imread(image_path)
    if bg_img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    bg_img = cv2.resize(bg_img, field_size)
    height, width, _ = bg_img.shape

    # output_json_path = os.path.splitext(json_path)[0] + "_merged.json"

    merged_tracks = load_and_merge_tracks(
        json_path=json_path,
        field_size=field_size,
        min_track_length=min_track_length,
        smoothing_window=smoothing_window,
        polyorder=polyorder,
        max_merge_gap=max_merge_gap,
        max_merge_distance=max_merge_distance,
        max_merge_overlap_frames=max_merge_overlap_frames,
        window_size=window_size,
        threshold=threshold,
        max_step=max_step,
    )

    merged_tracks = remove_referee_near_boundary(
        merged_tracks,
        field_size=field_size,
        margin_meter=30
    )

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
    min_frame = min(t["frame_range"][0] for t in merged_tracks)
    max_frame = max(t["frame_range"][1] for t in merged_tracks)

    for f in range(min_frame, max_frame + 1):
        frame_img = bg_img.copy()
        for track in merged_tracks:
            start_frame, end_frame = track["frame_range"]
            if not (start_frame <= f <= end_frame):
                continue
            index = f - start_frame
            if index < 0 or index >= len(track["points"]):
                continue

            points = track["points"][index]
            if points is None:
                continue
            x, y = int(points[0]), field_size[1] - int(points[1])
            color = team_colors.get(track["team"], (128, 128, 128))
            cv2.circle(frame_img, (x, y), 5, color, -1)
            cv2.putText(frame_img, str(track["track_id"]), (x + 6, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        writer.write(frame_img)

    # # save to output JSON
    # save_tracks_to_json(merged_tracks, output_json_path)

    writer.release()
    print(f"✅ Saved video to: {output_video_path}")


if __name__ == "__main__":
    start = time.time()
    # Example usage
    create_current_dot_video(
        json_path="./runs/detect/test_4k2/team_tracking.json",
        image_path="./data/images/mongkok_football_field.png",
        output_video_path="./trajectory_current_only.mp4",
        field_size=(1060, 660),
        min_track_length=10,
        smoothing_window=90,
        polyorder=7,
        max_step=20,
        max_merge_gap=20,
        max_merge_overlap_frames=15,
        max_merge_distance=100,
        window_size=20,
        threshold=0.9,
        fps=30
    )
    end = time.time()
    print(f"Execution time: {end - start:.2f} seconds")
