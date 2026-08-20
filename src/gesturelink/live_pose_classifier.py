"""
Running GestureLink's trained ML model in real time.

Pipeline:

Camera
   ↓
MediaPipe Hand Detection
   ↓
21 Hand Landmarks
   ↓
Feature Engineering
   ↓
73 Features
   ↓
Random Forest Classifier
   ↓
Gesture Prediction
"""

from pathlib import Path
import time

import cv2
import joblib
import mediapipe as mp
import pandas as pd

from feature_engineering import extract_features


# Setting the default laptop webcam.
DEFAULT_CAMERA_INDEX = 0

# Naming the OpenCV display window.
WINDOW_NAME = "GestureLink - ML Classifier"


# Finding the project root directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Locating the trained MediaPipe hand detector.
HAND_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "hand_landmarker.task"
)


# Locating our trained Random Forest model.
CLASSIFIER_PATH = (
    PROJECT_ROOT
    / "models"
    / "pose_classifier.joblib"
)


def load_classifier():
    """
    Loading the trained GestureLink classifier.
    """

    # Checking whether the trained model exists.
    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(
            f"Classifier not found at {CLASSIFIER_PATH}"
        )

    # Loading the saved model bundle.
    model_bundle = joblib.load(
        CLASSIFIER_PATH
    )

    # Extracting the actual ML model.
    model = model_bundle["model"]

    # Extracting the feature order used during training.
    feature_columns = model_bundle[
        "feature_columns"
    ]

    # Extracting the model name for displaying.
    model_name = model_bundle[
        "model_name"
    ]

    return (
        model,
        feature_columns,
        model_name,
    )


def predict_gesture(
    landmarks,
    model,
    feature_columns,
):
    """
    Predicting the current hand gesture.
    """

    # Converting MediaPipe landmarks into the same
    # 73 features used during training.
    features = extract_features(
        landmarks
    )

    # Creating a DataFrame because sklearn
    # expects the same feature structure.
    feature_frame = pd.DataFrame(
        [features],
        columns=feature_columns,
    )

    # Asking the Random Forest model
    # to classify the gesture.
    prediction = model.predict(
        feature_frame
    )[0]


    confidence = None

    # Checking whether the model supports probability output.
    if hasattr(
        model,
        "predict_proba",
    ):

        probabilities = (
            model.predict_proba(
                feature_frame
            )[0]
        )

        predicted_index = list(
            model.classes_
        ).index(
            prediction
        )

        confidence = probabilities[
            predicted_index
        ]


    return (
        prediction,
        confidence,
    )


def draw_landmarks(
    frame,
    landmarks,
):
    """
    Drawing MediaPipe landmarks on the webcam frame.
    """

    height, width, _ = frame.shape


    # Converting normalized MediaPipe coordinates
    # into actual pixel positions.
    points = [
        (
            int(
                landmark.x * width
            ),
            int(
                landmark.y * height
            ),
        )
        for landmark in landmarks
    ]


    # Drawing connections between hand joints.
    for connection in (
        mp.tasks.vision
        .HandLandmarksConnections
        .HAND_CONNECTIONS
    ):

        cv2.line(
            frame,
            points[connection.start],
            points[connection.end],
            (60,160,60),
            2,
        )


    # Drawing all 21 landmarks.
    for point in points:

        cv2.circle(
            frame,
            point,
            4,
            (40,40,220),
            -1,
        )


def run_live_classifier():

    """
    Running real-time GestureLink ML recognition.
    """

    (
        model,
        feature_columns,
        model_name,
    ) = load_classifier()


    print(
        f"Loaded model: {model_name}"
    )


    # Opening the laptop webcam.
    camera = cv2.VideoCapture(
        DEFAULT_CAMERA_INDEX
    )


    if not camera.isOpened():
        raise RuntimeError(
            "Could not open webcam."
        )


    # Loading MediaPipe model settings.
    base_options = mp.tasks.BaseOptions(
        model_asset_path=str(
            HAND_MODEL_PATH
        )
    )


    options = (
        mp.tasks.vision
        .HandLandmarkerOptions(
            base_options=base_options,
            running_mode=(
                mp.tasks.vision
                .RunningMode.VIDEO
            ),
            num_hands=1,
        )
    )


    print(
        "GestureLink ML classifier started."
    )

    print(
        "Press E to exit."
    )


    try:

        with (
            mp.tasks.vision
            .HandLandmarker
            .create_from_options(
                options
            )
        ) as landmarker:


            while True:


                frame_received, frame = (
                    camera.read()
                )


                if not frame_received:
                    break


                # Mirroring webcam view.
                frame = cv2.flip(
                    frame,
                    1,
                )


                # Converting BGR to RGB.
                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )


                media_pipe_image = mp.Image(
                    image_format=(
                        mp.ImageFormat.SRGB
                    ),
                    data=rgb_frame,
                )


                timestamp_ms = int(
                    time.monotonic()
                    * 1000
                )


                # Detecting hand landmarks.
                result = (
                    landmarker
                    .detect_for_video(
                        media_pipe_image,
                        timestamp_ms,
                    )
                )


                if result.hand_landmarks:


                    landmarks = (
                        result
                        .hand_landmarks[0]
                    )


                    draw_landmarks(
                        frame,
                        landmarks,
                    )


                    prediction, confidence = (
                        predict_gesture(
                            landmarks,
                            model,
                            feature_columns,
                        )
                    )


                    label = (
                        prediction
                        .replace("_"," ")
                        .title()
                    )


                    cv2.putText(
                        frame,
                        f"Gesture: {label}",
                        (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (30,30,30),
                        2,
                    )


                    if confidence:

                        cv2.putText(
                            frame,
                            (
                                f"Confidence: "
                                f"{confidence:.2f}"
                            ),
                            (20,75),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (30,30,30),
                            2,
                        )


                else:

                    cv2.putText(
                        frame,
                        "No hand detected",
                        (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (30,30,30),
                        2,
                    )

                cv2.putText(
                    frame,
                    "Press E to exit",
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (30, 30, 30),
                    2,
                )

                cv2.imshow(
                    WINDOW_NAME,
                    frame,
                )


                key = (
                    cv2.waitKey(1)
                    & 0xFF
                )


                if key in (
                    ord("e"),
                    ord("E"),
                ):
                    break


    finally:

        # Releasing the webcam.
        camera.release()

        # Closing OpenCV windows.
        cv2.destroyAllWindows()

        print(
            "GestureLink ML closed safely."
        )



if __name__ == "__main__":
    run_live_classifier()