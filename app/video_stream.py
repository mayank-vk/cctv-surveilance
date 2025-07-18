import cv2
import time
from collections import deque
from model_interface import predict_frame
from notifier import send_alert
import os

# Folder to save the recorded clips

CLIP_SAVE_DIR = "saved_clips"
os.makedirs(CLIP_SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture("C:/Users/mayan/Downloads/test-video-2.avi")
prediction_buffer = deque(maxlen=15)

saving = False
video_writer = None
frames_saved = 0
max_frames_to_save = 25  # approx 5 second clip

fourcc = cv2.VideoWriter_fourcc(*'XVID')

# Cooldown between alerts
last_alert_time = 0
alert_cooldown_seconds = 15  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    pred = predict_frame(frame)
    is_violence = 1 if pred > 0.5 else 0
    prediction_buffer.append(is_violence)

    violence_count = prediction_buffer.count(1)
    label = "Violence" if violence_count >= 10 else "Normal"

    # Start saving only if cooldown period is over
    now = time.time()
    if label == "Violence" and not saving and (now - last_alert_time) > alert_cooldown_seconds:
        print("[!] Violence detected. Starting to save clip.")
        saving = True
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        clip_path = os.path.join(CLIP_SAVE_DIR, f"violence_clip_{timestamp}.avi")
        video_writer = cv2.VideoWriter(clip_path, fourcc, 5.0, (frame.shape[1], frame.shape[0]))
        frames_saved = 0
        last_alert_time = now  # Reset cooldown

    # Save frames
    if saving:
        video_writer.write(frame)
        frames_saved += 1
        if frames_saved >= max_frames_to_save:
            saving = False
            video_writer.release()
            send_alert(clip_path=clip_path, timestamp=timestamp)
            print("[✓] Clip saved successfully. Alert sent.")

    # Show label
    color = (0, 0, 255) if label == "Violence" else (0, 255, 0)
    cv2.putText(frame, f"{label} ({violence_count}/15)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Live Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
