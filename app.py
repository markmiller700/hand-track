import os
import time
import urllib.request
from pathlib import Path

import cv2

project_cache_dir = Path(__file__).resolve().parent / ".cache" / "matplotlib"
project_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(project_cache_dir))

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
except ImportError:
    raise SystemExit(
        "mediapipe is not installed.\n"
        "Use the project virtual environment:\n"
        "1. python -m venv .venv\n"
        "2. .venv/bin/python -m pip install -r requirements.txt\n"
        "3. .venv/bin/python app.py"
    )

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parent / "hand_landmarker.task"

# Landmark index for fingertips
INDEX_TIP = 8


def ensure_model():
    if not MODEL_PATH.exists():
        print("Downloading hand landmarker model (~5 MB)…")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")


# Draw connections between landmarks manually (Tasks API has no built-in draw_landmarks)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),       # thumb
    (0,5),(5,6),(6,7),(7,8),       # index
    (0,9),(9,10),(10,11),(11,12),  # middle
    (0,13),(13,14),(14,15),(15,16),# ring
    (0,17),(17,18),(18,19),(19,20),# pinky
    (5,9),(9,13),(13,17),          # palm
]


def draw_hand(frame, landmark_points):
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, landmark_points[start], landmark_points[end], (0, 200, 100), 2)
    for pt in landmark_points:
        cv2.circle(frame, pt, 4, (255, 255, 255), cv2.FILLED)
        cv2.circle(frame, pt, 4, (0, 180, 90), 1)


class HandTracker:
    def __init__(
        self,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    ):
        ensure_model()
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def process(self, frame):
        frame_height, frame_width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self._timestamp_ms += 1
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self.landmarker.detect_for_video(mp_image, self._timestamp_ms)

        if not results.hand_landmarks:
            return frame, []

        hands_info = []

        for i, hand_landmarks in enumerate(results.hand_landmarks):
            landmark_points = [
                (int(lm.x * frame_width), int(lm.y * frame_height))
                for lm in hand_landmarks
            ]

            draw_hand(frame, landmark_points)

            label = results.handedness[i][0].category_name
            score = results.handedness[i][0].score
            hands_info.append({"label": label, "score": score, "landmarks": landmark_points})

            # Label near wrist
            wrist_x, wrist_y = landmark_points[0]
            cv2.putText(
                frame, f"{label} {score:.2f}",
                (wrist_x + 10, wrist_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )

            # Index fingertip highlight + coords
            tip_x, tip_y = landmark_points[INDEX_TIP]
            cv2.circle(frame, (tip_x, tip_y), 10, (0, 0, 255), cv2.FILLED)
            cv2.putText(
                frame, f"Index: {tip_x}, {tip_y}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )

        return frame, hands_info

    def close(self):
        self.landmarker.close()


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise SystemExit("Could not open webcam. Check your camera connection.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    tracker = HandTracker()
    previous_time = time.time()

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to read frame from webcam.")
                break

            frame = cv2.flip(frame, 1)
            tracked_frame, hands_info = tracker.process(frame)

            current_time = time.time()
            fps = 1 / max(current_time - previous_time, 1e-6)
            previous_time = current_time

            cv2.putText(
                tracked_frame, f"FPS: {int(fps)}",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
            )

            if not hands_info:
                cv2.putText(
                    tracked_frame, "Show your hand to the camera",
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2,
                )

            cv2.imshow("Hand-Tracking", tracked_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
