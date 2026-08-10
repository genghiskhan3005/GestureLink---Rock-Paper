"""
This file basically does 4 jobs :
1. Finds the MediaPipe hand model
2. Opens the camera
3. Detects one hand and its 21 landmarks
4. Draws the landmarks and shows the result
"""
from pathlib import Path    # Helps python work with file paths     # Better than manually writing a path as the project can run on different computers 
import time # This is used to create timestamps, MediaPipe VIDEO mode requires each frame to have an increasing timestamp, otherwise throws an error as it means out of order frames

import cv2
import mediapipe as mp  # Detects hand and tracks landmarks

from pose_rules import classify_pose


DEFAULT_CAMERA_INDEX = 0
WINDOW_NAME = "Hand Landmarks"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Path(__file__) turns the path into a Path object
# .resolve() gets its complete absolute location
# .parents[0] -> gives the immediate directory of this .py file (ex. gesturelink/)
# parents[1] -> gives the parent directory of parents[0] (ex. src/)
# parents[2] -> gives the parent directory of parents[1] (ex. GestureLink/)
MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
# When using Path, / combines folders
# MediaPipe needs this .task to recognize hands


def draw_hand_landmarks(
    frame,
    detection_result,
) -> None:
    # Draws detected landmarks, connections and handedness on a frame

    frame_height, frame_width, _ = frame.shape # _ means we don't need that value (returns the number of color channels in this case, 3 -> red, blue, green)

    for hand_index, hand_landmarks in enumerate(
        detection_result.hand_landmarks # List of detected hands
    ):

        pose = classify_pose(hand_landmarks)

        # Converts MediaPipe's normalized coordinates into pixel coordinates. MediaPipe returns coordinates like 0.5 and 0.4 (ranging from 0 to 1). This function converts them into actual positions of where to draw something
        pixel_points = [
            (
                int(landmark.x * frame_width),
                int(landmark.y * frame_height),
            )
            for landmark in hand_landmarks
        ]

        # Draws the lines that connect the hand landmarks. MediaPipe already knows which points connect to which ones.
        for connection in (
            mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
        ):
            start_point = pixel_points[connection.start]    # Gets the pixel coordinates for the starting landmark
            end_point = pixel_points[connection.end]    # Gets the pixel coordinates for the ending landmark

            cv2.line(
                frame,
                start_point,
                end_point,
                (60, 160, 60),  # Line color between the landmarks (greenish color)
                2,  # Line thickness in pixels
            )

        # Draws one circle for each of the 21 landmarks.
        for point in pixel_points:
            cv2.circle(
                frame,
                point,
                4,  # Radius of the circle
                (40, 40, 220),
                thickness=-1, # -1 fills the circle completely
            )

        # MediaPipe also returns whether it sees a left or right hand.
        if hand_index < len(detection_result.handedness):
            handedness = detection_result.handedness[hand_index][0]
            hand_name = handedness.category_name
            if hand_name == "Left":
                hand_name = "Right"
            elif hand_name == "Right":
                hand_name = "Left"
            confidence = handedness.score

            label = f"{hand_name} hand: {confidence:.2f}"

            pose_label = pose.replace("_", " ").title()

            cv2.putText(
                frame,
                f"Pose: {pose_label}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (30, 30, 30),
                2,
                cv2.LINE_AA,
            )

            label_x = min(point[0] for point in pixel_points)   # Finds the smallest x-coordinate in the hand (leftmost part)
            label_y = max(
                30,
                min(point[1] for point in pixel_points) - 15,   # Finds the highest point of the hand   # Smaller y means higher on the screen for computers    # -15 puts the text slightly above the hand
            )   # max prevents the text from being placed too close to or outside the top edge

            cv2.putText(
                frame,  # image to draw on
                label,  # text
                (label_x, label_y), # text location
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,   # font size
                (30, 30, 30),   # dark text
                2,  # thickness
                cv2.LINE_AA,    # smoother text edges
            )


def run_hand_landmarker() -> None:  # This is the main function that runs the camera and the detector
    # Opens the webcam and tracks one hand in real time.

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"MediaPipe model was not found at: {MODEL_PATH}\n"
            "Download hand_landmarker.task into the models folder."
        )

    camera = cv2.VideoCapture(DEFAULT_CAMERA_INDEX)

    if not camera.isOpened():
        raise RuntimeError(
            "The webcam could not be opened. "
            "Close other camera applications and try again."
        )

    base_options = mp.tasks.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    )   # Gives MediaPipe the location of hand_landmarker.task

    options = mp.tasks.vision.HandLandmarkerOptions( # Deciding how the detector behaves
        base_options = base_options, # Using the model that has just been specified
        running_mode = mp.tasks.vision.RunningMode.VIDEO, # VIDEO is the running mode
        num_hands = 1, # This project needs only 1 hand
        min_hand_detection_confidence = 0.5,
        min_hand_presence_confidence = 0.5, # Once MediaPipe is processing the image, this controls how confident it needs to be that a hand is still present.
        min_tracking_confidence = 0.5,
    )

    print("GestureLink hand tracking started.")
    print("Show your hand to the camera.")
    print("Press E to exit.")

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(
            options
        ) as landmarker:    # Creates the actual hand tracking object, the AI detector
            while True: # Infinite camera loop
                frame_received, frame = camera.read()

                if not frame_received:
                    print("A webcam frame could not be read.")
                    break

                frame = cv2.flip(frame, 1)

                # OpenCV uses BGR, while MediaPipe expects RGB.
                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )   # Converts BGR to RGB, important because OpenCV(BGR) and MediaPipe(RGB) use different color formats

                media_pipe_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )

                timestamp_ms = int(time.monotonic() * 1000) # Returns continuously increasing time in seconds   # *1000 converts seconds into miliseconds

                detection_result = landmarker.detect_for_video( # Detection step -> Tells MediaPipe to analyze this frame at this timestamp
                    media_pipe_image,
                    timestamp_ms,
                )

                draw_hand_landmarks(
                    frame,
                    detection_result,
                )

                if not detection_result.hand_landmarks:
                    cv2.putText(
                        frame,
                        "No hand detected",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (30, 30, 30),
                        2,
                        cv2.LINE_AA,
                    )

                cv2.imshow(WINDOW_NAME, frame)

                pressed_key = cv2.waitKey(1) & 0xFF # Waits approximately 1 millisecond for a keyboard press

                if pressed_key in (ord("e"), ord("E")):
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Hand tracking closed safely.")


if __name__ == "__main__":  # Run run_hand_landmarker() only if this file is executed directly
    run_hand_landmarker()