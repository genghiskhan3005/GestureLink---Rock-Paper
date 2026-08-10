# Collecting labelled MediaPipe hand landmarks for GestureLink.
# When recording is active, we'll save one sample every few frames rather than every single frame.

import csv
from datetime import datetime  # Used to create a unique ID for each recording session.
from pathlib import Path
import time  # Used to generate timestamps for MediaPipe video processing.

import cv2
import mediapipe as mp

DEFAULT_CAMERA_INDEX = 0

WINDOW_NAME = "GestureLink - Dataset Collector"


# __file__ = this Python file.
# parents[0] = gesturelink
# parents[1] = src
# parents[2] = project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Location of Google's MediaPipe hand-landmark model.
MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"

# Where our collected raw ML data will be saved.
DATA_PATH = (PROJECT_ROOT/"data"/"raw"/"hand_landmarks.csv")

# We do not want to save every video frame because consecutive frames are almost identical.
# Instead, we save one sample every 5 frames.
SAVE_EVERY_N_FRAMES = 5


# Keyboard controls used while collecting data :
# Key "1" means we are recording open-palm examples.
# Key "2" means we are recording fist examples.
# Key "3" means we are recording other/random hand poses.
VALID_LABELS = {
    ord("1"): "open_palm",
    ord("2"): "fist",
    ord("3"): "other",
}


def create_csv_header() -> list[str]:
    columns = [
        "session_id",
        "user_id",
        "timestamp_ms",
        "label",
        "handedness",
    ]

    # MediaPipe gives us 21 landmarks.
    # Each landmark has x, y, and z coordinates.
    for landmark_index in range(21):
        columns.extend(
            [
                f"x{landmark_index}",
                f"y{landmark_index}",
                f"z{landmark_index}",
            ]
        )

    return columns


def save_sample(
    writer: csv.writer,
    session_id: str,
    user_id: str,
    timestamp_ms: int,
    label: str,
    handedness: str,
    landmarks,
) -> None:
    # Saves one labelled hand-landmark sample to the CSV file.

    # Starts the row with metadata.
    row = [
        session_id,
        user_id,
        timestamp_ms,
        label,
        handedness,
    ]

    # Adding the x, y, z coordinates of all 21 landmarks.
    for landmark in landmarks:
        row.extend(
            [
                landmark.x,
                landmark.y,
                landmark.z,
            ]
        )

    # Saves the complete row into the CSV file.
    writer.writerow(row)


def correct_handedness(detected_hand: str) -> str:
    if detected_hand == "Left":
        return "Right"

    if detected_hand == "Right":
        return "Left"

    return "Unknown"


def draw_status(
    frame,
    active_label: str | None,
    saved_samples: int,
) -> None:
    # Draws recording information on the webcam frame.

    # If no label is currently active, we are paused.
    status = (
        active_label
        if active_label is not None
        else "PAUSED"
    )

    # Shows what class is currently being recorded.
    cv2.putText(
        frame,
        f"Recording: {status}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )

    # Shows how many samples have been saved during this session.
    cv2.putText(
        frame,
        f"Samples saved: {saved_samples}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )

    # Displays the keyboard controls.
    cv2.putText(
        frame,
        "1 Palm | 2 Fist | 3 Other | 0 Pause | E Exit",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )


def run_dataset_collector() -> None:
# Run the webcam-based dataset collector

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"MediaPipe model was not found at:\n{MODEL_PATH}"
        )

    # Create data/raw automatically if it does not exist.
    DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Asks who is providing the training samples.
    # Later this lets us test whether our model works on unseen users.
    user_id = input(
        "Enter user ID (example: user_01): "
    ).strip()

    # Will use a default ID if the user presses Enter without typing anything.
    if not user_id:
        user_id = "user_01"

    # Every time we start the collector, a new session ID will be created
    # Example: 20260810_213501
    session_id = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    # Opens the laptop's default webcam.
    camera = cv2.VideoCapture(
        DEFAULT_CAMERA_INDEX
    )

    if not camera.isOpened():
        raise RuntimeError(
            "The webcam could not be opened."
        )

    # Tells MediaPipe where its pretrained hand model is stored.
    base_options = mp.tasks.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    )

    # Configuring MediaPipe for real-time video.
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,

        # VIDEO mode lets MediaPipe use tracking between frames.
        running_mode=mp.tasks.vision.RunningMode.VIDEO,

        # We only need one hand for the first GestureLink version.
        num_hands=1,

        # Minimum confidence required to detect a hand.
        min_hand_detection_confidence=0.5,

        # Minimum confidence required to believe the hand is still present.
        min_hand_presence_confidence=0.5,

        # Minimum confidence used while tracking the detected hand.
        min_tracking_confidence=0.5,
    )

    # Checks whether the dataset already exists.
    # If it does not, we need to write the CSV header first.
    file_exists = DATA_PATH.exists()

    # None means we are not currently recording any class.
    active_label: str | None = None

    # Counts webcam frames.
    frame_counter = 0

    # Counts how many ML samples were actually stored.
    saved_samples = 0

    print()
    print("GestureLink Dataset Collector")
    print("-----------------------------")
    print("1 = Open Palm")
    print("2 = Fist")
    print("3 = Other")
    print("0 = Pause")
    print("E = Exit")
    print()

    try:
        # Opens the CSV in append mode.
        # "a" means new samples are added rather than deleting old data.
        with DATA_PATH.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as csv_file:

            # Creates a CSV writer object.
            writer = csv.writer(csv_file)

            # Writes the header only when the file is first created.
            if not file_exists:
                writer.writerow(
                    create_csv_header()
                )

            # Creates the MediaPipe hand detector.
            with mp.tasks.vision.HandLandmarker.create_from_options(
                options
            ) as landmarker:

                # Main webcam loop.
                while True:

                    # Reads one frame from the webcam.
                    success, frame = camera.read()

                    if not success:
                        print(
                            "Could not read a frame from the webcam."
                        )
                        break

                    frame = cv2.flip(
                        frame,
                        1,
                    )

                    # OpenCV uses BGR colors.
                    # MediaPipe expects RGB.
                    rgb_frame = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB,
                    )

                    # Converting the NumPy/OpenCV image into a MediaPipe image.
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb_frame,
                    )

                    # MediaPipe VIDEO mode requires increasing timestamps.
                    timestamp_ms = int(
                        time.monotonic() * 1000
                    )

                    # Detecting the hand and its 21 landmarks.
                    result = landmarker.detect_for_video(
                        mp_image,
                        timestamp_ms,
                    )

                    frame_counter += 1

                    # Only continuing if a hand was detected.
                    if result.hand_landmarks:

                        # Since num_hands=1, using the first detected hand.
                        landmarks = result.hand_landmarks[0]

                        # Default value in case handedness is unavailable.
                        handedness = "Unknown"

                        # Checking whether MediaPipe returned left/right information.
                        if result.handedness:

                            detected_hand = (
                                result.handedness[0][0]
                                .category_name
                            )

                            # Fixes the left/right label because our frame is mirrored.
                            handedness = correct_handedness(
                                detected_hand
                            )

                        # Will save only when:
                        # 1. Recording is active.
                        # 2. The current frame matches our sampling interval.
                        should_save = (
                            active_label is not None
                            and frame_counter
                            % SAVE_EVERY_N_FRAMES
                            == 0
                        )

                        if should_save:
                            save_sample(
                                writer=writer,
                                session_id=session_id,
                                user_id=user_id,
                                timestamp_ms=timestamp_ms,
                                label=active_label,
                                handedness=handedness,
                                landmarks=landmarks,
                            )

                            # Immediately writes data to disk.
                            # This helps protect our samples if the program crashes.
                            csv_file.flush()

                            # Updates the session counter.
                            saved_samples += 1

                    # Drawing the current recording state onto the webcam.
                    draw_status(
                        frame,
                        active_label,
                        saved_samples,
                    )

                    # Showing the webcam frame.
                    cv2.imshow(
                        WINDOW_NAME,
                        frame,
                    )

                    # Waits 1 ms and check whether a key was pressed.
                    key = cv2.waitKey(1) & 0xFF

                    # If 1, 2, or 3 is pressed, starts recording the corresponding class.
                    if key in VALID_LABELS:

                        active_label = VALID_LABELS[key]

                        print(
                            f"Recording: {active_label}"
                        )

                    # Pressing 0 temporarily stops recording.
                    elif key == ord("0"):

                        active_label = None

                        print(
                            "Recording paused."
                        )

                    # Pressing E or e stops the program.
                    elif key in (
                        ord("e"),
                        ord("E"),
                    ):
                        break

    finally:
        camera.release()
        cv2.destroyAllWindows()

        # Printing a useful summary of the collection session.
        print()
        print(
            f"Session completed: {session_id}"
        )
        print(
            f"Samples saved: {saved_samples}"
        )
        print(
            f"Dataset location: {DATA_PATH}"
        )


if __name__ == "__main__":
    run_dataset_collector()