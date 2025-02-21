import subprocess
import json
from datetime import datetime, timedelta

def get_mediainfo_datetime(video_path):
    """ Run mediainfo and return parsed JSON, with debugging output """
    cmd = ["mediainfo", "--Output=JSON", video_path]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ Error running mediainfo: {result.stderr}")
        return None

    # print("🛠 Raw JSON Output from mediainfo:")
    # print(result.stdout)  # Debugging: Show JSON before parsing

    try:
        metadata = json.loads(result.stdout)
        return metadata
    except json.JSONDecodeError as e:
        print(f"❌ JSON Decode Error: {e}")
        return None


def get_video_start_time(metadata):
    """ Extracts Encoded_Date from metadata, checking 'media' key first """
    if not metadata or "media" not in metadata or "track" not in metadata["media"]:
        print("❌ Error: No 'track' key found in metadata. Checking full JSON structure:")
        # print(json.dumps(metadata, indent=4))  # Print full JSON for debugging
        return None

    try:
        for track in metadata["media"]["track"]:  # Now correctly accessing 'track'
            if track.get("@type") == "General":
                encoded_date = track.get("Encoded_Date")
                if encoded_date:
                    encoded_date = encoded_date.replace("UTC ", "").strip()
                    return datetime.strptime(encoded_date, "%Y-%m-%d %H:%M:%S")

    except Exception as e:
        print(f"⚠️ Error extracting Encoded_Date: {e}")

    print("❌ Error: Encoded_Date not found.")
    return None


def get_video_fps(metadata):
    """ Extracts FrameRate from metadata, handling 'media' structure """
    if not metadata or "media" not in metadata or "track" not in metadata["media"]:
        print("❌ Error: No 'track' key found in metadata for FPS extraction.")
        return None

    try:
        for track in metadata["media"]["track"]:
            if track.get("@type") == "General":
                frame_rate = track.get("FrameRate")
                if frame_rate:
                    return float(frame_rate)
    except Exception as e:
        print(f"⚠️ Error extracting FPS: {e}")

    print("❌ Error: FrameRate not found in metadata.")
    return None


def get_frame_timestamp(video_path, frame_number, utc_offset=0):
    """ Compute real timestamp for a given frame """
    metadata = get_mediainfo_datetime(video_path)

    if not metadata:
        print("❌ Error: No metadata extracted.")
        return None

    start_time = get_video_start_time(metadata)
    fps = get_video_fps(metadata)

    if start_time is None:
        print("❌ Error: Could not determine video start time.")
        return None
    if fps is None:
        print("❌ Error: Could not determine FPS.")
        return None

    frame_time = timedelta(seconds=(frame_number / fps))
    adjusted_time = start_time + frame_time + timedelta(hours=utc_offset)
    return adjusted_time

# Example Usage
video_file = "./data/video/GX040011.mp4"
frame_num = 10000  # Replace with desired frame number
utc_offset = 8  # Example: UTC+8 timezon

real_timestamp = get_frame_timestamp(video_file, frame_num, utc_offset)

if real_timestamp:
    print(f"🕒 Frame {frame_num} was recorded at: {real_timestamp}")
