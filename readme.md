# 🛡️ Real-Time Violence Detection System Using CCTV Footage

This project is a **real-time violence detection system** that processes live or recorded CCTV video feeds and raises alerts when violent activity is detected. It uses **deep learning**, **OpenCV**, and a **Flask-based UI** to stream video, visualize predictions, and send alerts with saved video clips.

---

## 📁 Project Structure

cctv-surveillance-project/
│
├── app/
│ ├── model_interface.py # Loads trained model & predicts on each frame
│ ├── notifier.py # Sends alert & logs timestamp
│ ├── frame_dataset.py # Prepares dataset from extracted video frames
│ ├── train_model.py # Custom CNN training script using Keras
│
├── data/
│ └── processed_frames/ # Preprocessed frames used for model training
│
├── model/
│ └── violence_detection_model.h5 # Trained CNN model
│
├── saved_clips/
│ └── *.avi # Saved clips when violence is detected
│
├── ui/
│ ├── dashboard.py # Flask app (video + alert + clip download)
│ ├── templates/
│ │ ├── index.html # Live feed + alert section
│ │ └── saved_clips.html # Lists saved clips for download
│
└── README.md


---

## 🚀 Features

- 🔍 **Frame-by-frame analysis** with a trained CNN
- 🎯 **Buffer smoothing logic** to reduce false positives
- 🧠 Custom CNN trained on [RWF-2000 dataset](https://www.kaggle.com/datasets/mohamedhanyyy/rwf-2000-real-world-violence-dataset)
- ⚠️ **Automatic alerts** with saved 5-second clips
- 💻 **Web dashboard** using Flask: live video + alert status + clip download
- 📼 Supports both **webcam** and **test video files**

---

## 🧠 Model Training (Optional)

If you want to retrain:

1. Extract frames from raw dataset using `frame_dataset.py`
2. Place under `data/processed_frames/` structured as:
    ```
    processed_frames/
      ├── Violence/
      └── NonViolence/
    ```
3. Run model training:
    ```bash
    python app/train_model.py
    ```

Model will be saved to: `model/violence_detection_model.h5`

---

## ▶️ Running the System

1. Install requirements:
    ```bash
    pip install -r requirements.txt
    ```

2. Edit `VIDEO_PATH` in `ui/dashboard.py` to either:
    - Use webcam → `cv2.VideoCapture(0)`
    - Use test video → `cv2.VideoCapture("path/to/video.avi")`

3. Start Flask UI:
    ```bash
    python ui/dashboard.py
    ```

4. Open in browser:
    ```
    http://127.0.0.1:5000/
    ```

---

## 📩 Alerts and Saved Clips

- When violence is detected:
  - A **5-second clip** is saved under `saved_clips/`
  - The UI shows **alert message** with timestamp
  - Clip becomes downloadable from the UI

---

## 🧪 Testing Tips

- You can test using **YouTube fight clips** in fullscreen or
- Replace webcam with a test `.avi` file for consistent evaluation

---

## ✅ Future Enhancements

- Deploy to cloud (e.g., Streamlit, Heroku, Docker)
- Add SMS/Email alert system
- Improve model accuracy using transfer learning (e.g., ResNet, MobileNet)

---

## ✨ Credits

- Dataset: [RWF-2000 on Kaggle](https://www.kaggle.com/datasets/mohamedhanyyy/rwf-2000-real-world-violence-dataset)
- Libraries: OpenCV, TensorFlow/Keras, Flask

---

## 📷 Sample Screenshot

![UI Screenshot](path/to/screenshot.png) <!-- Add one if you want -->

---

## 💡 Author

**Mayank Kejriwal**  
Project built with 💻, ☕, and 🔍
