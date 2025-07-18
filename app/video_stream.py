import cv2
import time
from collections import deque
from model_interface import predict_frame
from notifier import send_alert
import os

#folder to save the recorded clips
CLIP_SAVE_DIR = "saved_clips"
os.makedirs(CLIP_SAVE_DIR, exist_ok=True)



cap = cv2.VideoCapture(0)
prediction_buffer = deque(maxlen=15)

saving = False
video_writer = None
frames_saved = 0
max_frames_to_save = 75  #appx 5 second clip

fourcc = cv2.VideoWriter_fourcc(*'XVID')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    pred = predict_frame(frame)
    prediction_buffer.append(pred)

    avg_pred = sum(prediction_buffer) / len(prediction_buffer)
    label = "Violence" if avg_pred > 0.5 else "Normal"

    # Start saving if violence is detected and not already saving
    if label == "Violence" and not saving:
        print("[!] Violence detected. Starting to save clip.")
        saving = True
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        clip_path = os.path.join(CLIP_SAVE_DIR, f"violence_clip_{timestamp}.avi")
        video_writer = cv2.VideoWriter(clip_path, fourcc, 15.0, (frame.shape[1], frame.shape[0]))
        frames_saved = 0

    # Save frames while saving flag is on
    if saving:
        video_writer.write(frame)
        frames_saved += 1
        if frames_saved >= max_frames_to_save:
            saving = False
            video_writer.release()
            send_alert(clip_path=clip_path,timestamp=timestamp)
            print("[✓] Clip saved successfully.Alert sent.")

    # Show label
    cv2.putText(frame, f"{label} ({avg_pred:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if label == "Violence" else (0, 255, 0), 2)
    cv2.imshow("Live Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
