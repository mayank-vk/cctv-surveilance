import os
import datetime

def send_alert(clip_path: str, timestamp: str):
    print("\n[ALERT] ⚠️ Violence detected!")
    print(f"[TIME]  📅 {timestamp}")
    print(f"[CLIP]  🎥 Saved at: {clip_path}\n")

    # log alerts to a file
    log_path = "alert_log.txt"
    with open(log_path, "a") as log_file:
        log_file.write(f"{timestamp} - Violence detected - Clip: {clip_path}\n")
