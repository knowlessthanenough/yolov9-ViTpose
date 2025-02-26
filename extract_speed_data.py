import csv
from datetime import datetime, timedelta, timezone

def find_max_speed_in_range(csv_file_path, reference_timestamp, time_buffer=60, csv_utc_offset=8):
    """
    Finds all speed data within ±time_buffer seconds of reference_timestamp (Unix time)
    from a CSV. Assumes the CSV times are in a certain UTC offset (csv_utc_offset).
    """

    # Convert reference_timestamp to float, in case it's passed as a string
    ref_ts = float(reference_timestamp)

    # Build a timezone object for the CSV’s offset (e.g., +8 => UTC+8)
    offset_delta = timedelta(hours=csv_utc_offset)
    csv_tz = timezone(offset_delta)

    max_speed = None

    with open(csv_file_path, 'r', newline='', encoding='utf-8') as f:
        # Skip the first three lines manually:
        for _ in range(3):
            next(f, None)

        reader = csv.DictReader(f)

        for row in reader:
            # Safety check: skip rows missing Date/Time
            if not row or 'Date' not in row or 'Time' not in row:
                continue

            date_val = row['Date']
            time_val = row['Time']
            if not date_val or not time_val:
                continue

            date_time_str = f"{date_val} {time_val}"

            # Parse using month/day/year + 12-hour time + AM/PM
            row_dt_naive = datetime.strptime(date_time_str, '%d/%m/%Y %I:%M:%S %p')

            # Attach the CSV's timezone
            row_dt = row_dt_naive.replace(tzinfo=csv_tz)

            # Convert that datetime to a Unix timestamp (UTC)
            row_ts = row_dt.timestamp()

            # Check if within ±time_buffer
            if abs(row_ts - ref_ts) <= time_buffer:
                speed_str = row.get('Speed', '')
                if speed_str:
                    speed_val = float(speed_str)
                    if max_speed is None or speed_val > max_speed:
                        max_speed = speed_val

    return max_speed


if __name__ == "__main__":
    csv_path = "./data/excel/PR_20250209_1537_0902_Jockey_Club.csv"
    ref_unix_ts = 1739086810  # Example
    result = find_max_speed_in_range(csv_path, ref_unix_ts, time_buffer=1000, csv_utc_offset=8)
    print("Max speed:", result)
