"""
Running GestureLink's trained ML pose classifier in real time.

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
Prediction Smoothing
    ↓
Stable Gesture Output
"""


from pathlib import Path
import time

import cv2
import joblib
import mediapipe as mp
import pandas as pd

from feature_engineering import extract_features
from prediction_smoother import PredictionSmoother


# Setting the default webcam index.
DEFAULT_CAMERA_INDEX = 0


# Naming the OpenCV display window.
WINDOW_NAME = "GestureLink - ML Pose Classifier"


# Finding the project root directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Locating the MediaPipe hand detection model.
HAND_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "hand_landmarker.task"
)


# Locating the trained Random Forest model.
CLASSIFIER_PATH = (
    PROJECT_ROOT
    / "models"
    / "pose_classifier.joblib"
)



def load_classifier():
    """
    Loading the trained ML classifier.
    """

    # Checking whether the trained model exists.
    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(
            f"Classifier not found at: {CLASSIFIER_PATH}"
        )


    # Loading the saved model bundle.
    model_bundle = joblib.load(
        CLASSIFIER_PATH
    )


    # Extracting the trained sklearn model.
    model = model_bundle["model"]


    # Extracting the feature order used during training.
    feature_columns = model_bundle[
        "feature_columns"
    ]


    # Extracting the model name.
    model_name = model_bundle[
        "model_name"
    ]


    return (
        model,
        feature_columns,
        model_name,
    )



def predict_pose(
    landmarks,
    model,
    feature_columns,
):
    """
    Predicting the current hand pose using
    the trained ML model.
    """


    # Converting MediaPipe landmarks into
    # the same 73 features used during training.
    features = extract_features(
        landmarks
    )


    # Creating a DataFrame with the same
    # feature order used during training.
    feature_frame = pd.DataFrame(
        [features],
        columns=feature_columns,
    )


    # Asking the trained model to classify
    # the current hand pose.
    prediction = model.predict(
        feature_frame
    )[0]


    confidence = None


    # Checking whether the model supports
    # probability prediction.
    if hasattr(
        model,
        "predict_proba",
    ):

        probabilities = (
            model.predict_proba(
                feature_frame
            )[0]
        )


        # Finding the probability of the
        # predicted class.
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



def draw_hand_landmarks(
    frame,
    landmarks,
):
    """
    Drawing hand landmarks and connections.
    """


    frame_height, frame_width, _ = (
        frame.shape
    )


    # Converting MediaPipe normalized
    # coordinates into screen pixels.
    pixel_points = [
        (
            int(
                landmark.x
                * frame_width
            ),
            int(
                landmark.y
                * frame_height
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
            pixel_points[
                connection.start
            ],
            pixel_points[
                connection.end
            ],
            (60, 160, 60),
            2,
        )


    # Drawing all 21 landmark points.
    for point in pixel_points:

        cv2.circle(
            frame,
            point,
            4,
            (40, 40, 220),
            -1,
        )



def run_live_pose_classifier():
    """
    Running GestureLink's complete
    live ML gesture recognition system.
    """


    (
        model,
        feature_columns,
        model_name,
    ) = load_classifier()


    smoother = PredictionSmoother(
    confirmation_frames=3
)


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
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
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


                # Reading a frame from webcam.
                frame_received, frame = (
                    camera.read()
                )


                if not frame_received:
                    break



                # Flipping the frame to create
                # a mirror-like webcam view.
                frame = cv2.flip(
                    frame,
                    1,
                )



                # Converting OpenCV BGR
                # into MediaPipe RGB format.
                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )



                # Creating a MediaPipe image.
                media_pipe_image = mp.Image(
                    image_format=(
                        mp.ImageFormat.SRGB
                    ),
                    data=rgb_frame,
                )



                # Creating increasing timestamps
                # required by MediaPipe VIDEO mode.
                timestamp_ms = int(
                    time.monotonic()
                    * 1000
                )



                # Detecting hand landmarks.
                detection_result = (
                    landmarker
                    .detect_for_video(
                        media_pipe_image,
                        timestamp_ms,
                    )
                )



                if detection_result.hand_landmarks:


                    # Taking the first detected hand.
                    landmarks = (
                        detection_result
                        .hand_landmarks[0]
                    )



                    draw_hand_landmarks(
                        frame,
                        landmarks,
                    )



                    prediction, confidence = (
                        predict_pose(
                            landmarks,
                            model,
                            feature_columns,
                        )
                    )



                    # Sending raw prediction
                    # into the smoother.
                    stable_prediction = (
                        smoother.update(
                            prediction
                        )
                    )



                    readable_prediction = (
                        stable_prediction
                        .replace("_", " ")
                        .title()
                    )



                    # Showing stable gesture.
                    cv2.putText(
                        frame,
                        (
                            f"Gesture: "
                            f"{readable_prediction}"
                        ),
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (30, 30, 30),
                        2,
                        cv2.LINE_AA,
                    )



                    if confidence is not None:

                        cv2.putText(
                            frame,
                            (
                                f"Confidence: "
                                f"{confidence:.2f}"
                            ),
                            (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (30, 30, 30),
                            2,
                            cv2.LINE_AA,
                        )



                else:

                    # Resetting history when no hand
                    # is being detected.
                    smoother.reset()


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



                # Showing model name.
                cv2.putText(
                    frame,
                    (
                        f"Model: "
                        f"{model_name}"
                    ),
                    (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (30, 30, 30),
                    2,
                    cv2.LINE_AA,
                )



                # Showing exit instruction.
                cv2.putText(
                    frame,
                    "Press E to exit",
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (30, 30, 30),
                    2,
                    cv2.LINE_AA,
                )



                # Displaying the final processed frame.
                cv2.imshow(
                    WINDOW_NAME,
                    frame,
                )



                pressed_key = (
                    cv2.waitKey(1)
                    & 0xFF
                )



                if pressed_key in (
                    ord("e"),
                    ord("E"),
                ):
                    break



    finally:

        # Releasing the webcam.
        camera.release()


        # Closing all OpenCV windows.
        cv2.destroyAllWindows()


        print(
            "GestureLink closed safely."
        )



if __name__ == "__main__":

    run_live_pose_classifier()