from flask import Flask, render_template, Response, jsonify, send_from_directory
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import os
import threading
import datetime
import time
from collections import deque
import app
from app.model_interface import predict_frame
from app.notifier import send_alert


app = Flask(__name__)

# Paths and constants
VIDEO_PATH = "C:/Users/mayan/Downloads/test-video.avi"
CLIP_SAVE_DIR = "saved_clips"
os.makedirs(CLIP_SAVE_DIR, exist_ok=True)

latest_alert = {"message": "No alert", "timestamp": ""}
prediction_buffer = deque(maxlen=15)
saving = False
video_writer = None
frames_saved = 0
max_frames_to_save = 75
fourcc = cv2.VideoWriter_fourcc(*'XVID')
last_alert_time = 0
alert_cooldown_seconds = 15
cap = cv2.VideoCapture(VIDEO_PATH)


def generate_frames():
    global saving, video_writer, frames_saved, last_alert_time

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Predict and update buffer
        pred = predict_frame(frame)
        is_violence = 1 if pred > 0.5 else 0
        prediction_buffer.append(is_violence)

        # Wait until buffer fills
        if len(prediction_buffer) < prediction_buffer.maxlen:
            label = "Normal"
            violence_count = prediction_buffer.count(1)
        else:
            violence_count = prediction_buffer.count(1)
            label = "Violence" if violence_count >= 10 else "Normal"

        # Save clip logic with cooldown
        now = time.time()
        if label == "Violence" and not saving and (now - last_alert_time) > alert_cooldown_seconds:
            saving = True
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            clip_path = os.path.join(CLIP_SAVE_DIR, f"violence_clip_{timestamp}.avi")
            video_writer = cv2.VideoWriter(clip_path, fourcc, 15.0, (frame.shape[1], frame.shape[0]))
            frames_saved = 0
            last_alert_time = now
            latest_alert["message"] = "⚠️ Violence Detected!"
            latest_alert["timestamp"] = timestamp

        if saving:
            video_writer.write(frame)
            frames_saved += 1
            if frames_saved >= max_frames_to_save:
                saving = False
                video_writer.release()
                send_alert(clip_path=clip_path, timestamp=timestamp)
                prediction_buffer.clear()  # Reset buffer after alert

        # Show label and overlay
        color = (0, 0, 255) if label == "Violence" else (0, 255, 0)
        cv2.putText(frame, f"{label} ({violence_count}/15)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')




@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/latest_alert')
def get_alert():
    return jsonify(latest_alert)


@app.route('/saved_clips')
def list_clips():
    clips_dir = os.path.join(os.getcwd(), 'saved_clips')
    clips = os.listdir(clips_dir) if os.path.exists(clips_dir) else []
    return render_template('saved_clips.html', clips=clips)


@app.route('/download_clip/<filename>')
def download_clip(filename):
    clips_dir = os.path.join(os.getcwd(), 'saved_clips')
    return send_from_directory(clips_dir, filename)


if __name__ == "__main__":
    app.run(debug=True)
