import threading
import time

latest_alert = {
    "message": None,
    "timestamp": None
}

def reset_alert_after_delay(delay=10):
    def _reset():
        time.sleep(delay)
        latest_alert["message"] = None
        latest_alert["timestamp"] = None
    threading.Thread(target=_reset, daemon=True).start()

def send_alert(clip_path, timestamp):
    global latest_alert
    latest_alert["message"] = f"Violence detected at {timestamp}"
    latest_alert["timestamp"] = timestamp
    print("[⚠] ALERT SENT:", latest_alert["message"])
    reset_alert_after_delay(delay=10)  # Reset after 10 sec
