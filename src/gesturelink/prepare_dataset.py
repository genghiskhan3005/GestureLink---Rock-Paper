"""Preparing the raw GestureLink landmark dataset for machine learning."""

from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import (
    center_landmarks,
    extract_joint_angles,
    normalize_scale,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "hand_landmarks.csv"
)

PROCESSED_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

FEATURES_PATH = (
    PROCESSED_DATA_DIR
    / "features.csv"
)


def row_to_landmark_array(row: pd.Series) -> np.ndarray:
    """Converting one CSV row into a 21 x 3 landmark array."""

    coordinates = []

    # Each row contains x0,y0,z0 ... x20,y20,z20.
    for landmark_index in range(21):
        coordinates.append(
            [
                row[f"x{landmark_index}"],
                row[f"y{landmark_index}"],
                row[f"z{landmark_index}"],
            ]
        )

    # Converting the Python list into a NumPy array.
    landmarks = np.array(
        coordinates,
        dtype=np.float64,
    )

    # Defensive check so malformed samples fail loudly.
    if landmarks.shape != (21, 3):
        raise ValueError(
            f"Expected shape (21, 3), received {landmarks.shape}"
        )

    return landmarks


def build_feature_names() -> list[str]:
    """Creating readable names for every generated ML feature."""

    feature_names = []

    # 63 normalized coordinate features.
    for landmark_index in range(21):
        feature_names.extend(
            [
                f"norm_x{landmark_index}",
                f"norm_y{landmark_index}",
                f"norm_z{landmark_index}",
            ]
        )

    # These correspond to the 10 angles in feature_engineering.py.
    feature_names.extend(
        [
            "thumb_mcp_angle",
            "thumb_ip_angle",
            "index_pip_angle",
            "index_dip_angle",
            "middle_pip_angle",
            "middle_dip_angle",
            "ring_pip_angle",
            "ring_dip_angle",
            "pinky_pip_angle",
            "pinky_dip_angle",
        ]
    )

    return feature_names


def extract_row_features(row: pd.Series) -> np.ndarray:
    """Converting one raw CSV sample into its final feature vector."""

    # Rebuilding the original 21 x 3 MediaPipe coordinates.
    landmarks = row_to_landmark_array(row)

    # Removing absolute screen position by making the wrist the origin.
    centered = center_landmarks(landmarks)

    # Reducing the effect of camera distance and physical hand size.
    normalized = normalize_scale(centered)

    # Converting the 21 x 3 matrix into 63 normalized coordinate features.
    normalized_coordinates = normalized.flatten()

    # Adding 10 finger-joint angle features.
    joint_angles = extract_joint_angles(normalized)

    # Final feature vector:
    # 63 normalized coordinates + 10 angles = 73 features.
    return np.concatenate(
        [
            normalized_coordinates,
            joint_angles,
        ]
    )


def prepare_dataset() -> None:
    """Converting raw GestureLink data into an ML-ready dataset."""

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at: {RAW_DATA_PATH}"
        )

    # Loading all 1,185 raw collected samples.
    raw_data = pd.read_csv(RAW_DATA_PATH)

    print()
    print("=" * 60)
    print("GESTURELINK DATASET PREPARATION")
    print("=" * 60)

    print(
        f"\nRaw samples: {len(raw_data)}"
    )

    # Generating a feature vector for every raw sample.
    feature_rows = []

    for _, row in raw_data.iterrows():
        features = extract_row_features(row)
        feature_rows.append(features)

    # Stacking every feature vector into one matrix.
    feature_matrix = np.vstack(feature_rows)

    print(
        f"Feature matrix shape: {feature_matrix.shape}"
    )

    # Expecting exactly 73 engineered features.
    if feature_matrix.shape[1] != 73:
        raise ValueError(
            f"Expected 73 features, "
            f"received {feature_matrix.shape[1]}"
        )

    feature_names = build_feature_names()

    # Converting the NumPy feature matrix into a DataFrame.
    processed_data = pd.DataFrame(
        feature_matrix,
        columns=feature_names,
    )

    # Preserving metadata required for grouped evaluation.
    processed_data.insert(
        0,
        "handedness",
        raw_data["handedness"].values,
    )

    processed_data.insert(
        0,
        "user_id",
        raw_data["user_id"].values,
    )

    processed_data.insert(
        0,
        "session_id",
        raw_data["session_id"].values,
    )

    # The target label is what our classifier will predict.
    processed_data.insert(
        0,
        "label",
        raw_data["label"].values,
    )

    # Creating data/processed automatically.
    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed_data.to_csv(
        FEATURES_PATH,
        index=False,
    )

    print(
        f"Processed samples: {len(processed_data)}"
    )

    print(
        f"Processed columns: {len(processed_data.columns)}"
    )

    print(
        f"Saved to: {FEATURES_PATH}"
    )

    print("\nClass distribution:")

    print(
        processed_data["label"]
        .value_counts()
    )

    print("\nUsers:")

    print(
        processed_data["user_id"]
        .value_counts()
    )

    print()
    print("=" * 60)


if __name__ == "__main__":
    prepare_dataset()