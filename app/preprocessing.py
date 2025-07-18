import os
import cv2

def extract_frames(video_path, output_dir, frame_rate=5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video: {video_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    frame_id = 0
    count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        if count % frame_rate == 0:
            frame = cv2.resize(frame, (224, 224))
            frame_filename = os.path.join(output_dir, f"frame_{frame_id:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_id += 1

        count += 1

    cap.release()


def process_dataset_from_local(video_dir, output_root=r"C:/Users/mayan/Desktop/cctv-surveilance-project/data/processed_frames", max_per_class=50):
    violence_path = os.path.join(video_dir, "Fight")
    non_violence_path = os.path.join(video_dir, "NonFight")

    violence_videos = sorted(os.listdir(violence_path))[:max_per_class]
    non_violence_videos = sorted(os.listdir(non_violence_path))[:max_per_class]

    for idx, video_file in enumerate(violence_videos):
        video_path = os.path.join(violence_path, video_file)
        output_dir = os.path.join(output_root, "violence", f"video_{idx:04d}")
        extract_frames(video_path, output_dir)
        print(f"[VIOLENCE] Processed {video_file}")

    for idx, video_file in enumerate(non_violence_videos):
        video_path = os.path.join(non_violence_path, video_file)
        output_dir = os.path.join(output_root, "non_violence", f"video_{idx:04d}")
        extract_frames(video_path, output_dir)
        print(f"[NON-VIOLENCE] Processed {video_file}")


if __name__ == "__main__":
    process_dataset_from_local(
        video_dir=r"C:/Users/mayan/Downloads/archive/RWF-2000/train",
        max_per_class=150
    )
