import json
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import defaultdict
import numpy as np
from scipy.signal import savgol_filter
import cv2

def draw_smoothed_paths_from_json_safe(json_path, image_path, field_size=(1060, 660), min_track_length=5, smoothing_window=7, polyorder=2):
    """
    Draw smoothed 2D trajectories from tracking JSON, skipping invalid team labels.

    Args:
        json_path (str): Path to the tracking JSON file.
        field_size (tuple): (width, height) of the field.
        min_track_length (int): Minimum number of projected points to consider.
        smoothing_window (int): Window size for Savitzky-Golay filter.
        polyorder (int): Polynomial order for Savitzky-Golay filter.
    """
    with open(json_path) as f:
        data = json.load(f)

    bg_img = cv2.imread(image_path)
    if bg_img is None:
        raise ValueError(f"Could not read image from {image_path}")
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
    bg_img = cv2.resize(bg_img, field_size)



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
    print("Team assignments by track_id:", team_votes)

    paths_by_team = defaultdict(list)
    for obj in data:
        tid = obj['track_id']
        proj = obj.get('projected', [])
        if len(proj) < min_track_length:
            continue

        xs, ys = zip(*[pt for pt in proj if pt is not None])
        xs, ys = np.array(xs), np.array(ys)

        in_bounds = (xs >= 0) & (xs <= field_size[0]) & (ys >= 0) & (ys <= field_size[1])
        if in_bounds.sum() < min_track_length:
            continue

        xs, ys = xs[in_bounds], ys[in_bounds]

        if len(xs) >= smoothing_window:
            xs = savgol_filter(xs, window_length=smoothing_window, polyorder=polyorder)
            ys = savgol_filter(ys, window_length=smoothing_window, polyorder=polyorder)

        if tid in team_votes:
            paths_by_team[team_votes[tid]].append((tid, xs, ys))
        else:
            paths_by_team['ball'].append((tid, xs, ys))

    fig, ax = plt.subplots(figsize=(12, 7))
    team_colors = {
        'eastern': 'blue',
        'easterngoalkeeper': 'green',
        'kitchee': 'pink',
        'kitcheegoalkeeper': 'orange',
        'referee': 'yellow',
        'ball': 'black',
    }
    # Set image as background with correct coordinate system
    ax.imshow(bg_img, extent=[0, field_size[0], 0, field_size[1]])  # y-axis flipped

    for team, tracks in paths_by_team.items():
        color = team_colors.get(team, 'gray')
        for tid, xs, ys in tracks:
            ax.plot(xs, ys, color=color, alpha=0.8)
            ax.scatter(xs[-1], ys[-1], color=color)
            ax.text(xs[-1], ys[-1], str(tid), fontsize=8, color='black')

    ax.set_xlim(0, field_size[0])
    ax.set_ylim(0, field_size[1])
    ax.set_title("2D Smoothed Player + Ball Trajectories")
    plt.show()

if  __name__ == "__main__":
    # Example usage
    draw_smoothed_paths_from_json_safe("./runs/detect/test_4k-crop-from-video/team_tracking.json", "./data/images/mongkok_football_field.png", smoothing_window= 30)
