# We want the ML classifier to learn hand shape, not screen position or hand size

"""
Raw landmarks
      ↓
1. Wrist centering
      ↓
2. Scale normalization
      ↓
3. Geometric angles
      ↓
Final feature vector
"""
import math
import numpy as np


NUM_LANDMARKS = 21  # MediaPipe provides exactly 21 landmarks per detected hand.
WRIST_INDEX = 0  # Landmark 0 represents the wrist.

def landmarks_to_array(landmarks) -> np.ndarray:
    # Converts MediaPipe landmarks into a 21 x 3 NumPy array.

    # Extract x, y, and z from each of MediaPipe's 21 landmarks.
    coordinates = np.array(
        [
            [landmark.x, landmark.y, landmark.z]
            for landmark in landmarks
        ],
        dtype=np.float64,
    )

    # Catching malformed input early instead of silently training on bad data.
    if coordinates.shape != (NUM_LANDMARKS, 3):
        raise ValueError(
            f"Expected landmark shape (21, 3), "
            f"but received {coordinates.shape}."
        )

    return coordinates


def center_landmarks(coordinates: np.ndarray) -> np.ndarray:
    # Translates landmarks so that the wrist becomes the origin.

    # Saves the wrist's x, y, z coordinates.
    wrist = coordinates[WRIST_INDEX]

    # Subtract the wrist from every landmark.
    centered = coordinates - wrist

    return centered

    # Now if we move our entire hand across the screen, the relative geometry stays approximately the same. -> Translation invariance


def normalize_scale(centered: np.ndarray) -> np.ndarray:
    # Normalizing landmark coordinates to reduce sensitivity to hand size so that it doesnt matter if the hand is close to camera or far from camera

    # Calculating the Euclidean distance from the wrist to every landmark.
    distances_from_wrist = np.linalg.norm(
        centered,
        axis=1,
    )

    # Using the farthest landmark as our hand-size reference.
    hand_scale = np.max(distances_from_wrist)

    # Protecting against division by zero for malformed landmarks.
    if hand_scale <= 1e-8:
        raise ValueError(
            "Cannot normalize landmarks with near-zero hand scale."
        )

    # Dividing the entire hand by one common scale factor.
    normalized = centered / hand_scale

    return normalized

    # Suppose one hand produces wrist → fingertip distance = 0.50
    # Another hand produces wrist → fingertip distance = 0.25
    # After scale normalization, both can become approximately 1.0


def calculate_angle(
    point_a: np.ndarray,
    point_b: np.ndarray,
    point_c: np.ndarray,
) -> float:
    # essentially the NumPy version of the geometry I already used in pose_rules.py

    # Create the two vectors that meet at point B.
    vector_ba = point_a - point_b
    vector_bc = point_c - point_b

    magnitude_ba = np.linalg.norm(vector_ba)
    magnitude_bc = np.linalg.norm(vector_bc)

    if magnitude_ba <= 1e-8 or magnitude_bc <= 1e-8:
        return 0.0

    cosine_angle = np.dot(
        vector_ba,
        vector_bc,
    ) / (
        magnitude_ba * magnitude_bc
    )

    # Numerical floating-point error can occasionally create values such as 1.00000001, which arccos cannot accept.
    cosine_angle = np.clip(
        cosine_angle,
        -1.0,
        1.0,
    )

    return float(
        np.degrees(
            np.arccos(cosine_angle)
        )
    )


def extract_joint_angles(
    coordinates: np.ndarray,
) -> np.ndarray:
    # Extracting important finger joint angles.

    # Each tuple represents:
    # (landmark before joint, joint landmark, landmark after joint)
    angle_triplets = [
        # Thumb
        (1, 2, 3),
        (2, 3, 4),

        # Index
        (5, 6, 7),
        (6, 7, 8),

        # Middle
        (9, 10, 11),
        (10, 11, 12),

        # Ring
        (13, 14, 15),
        (14, 15, 16),

        # Pinky
        (17, 18, 19),
        (18, 19, 20),
    ]

    angles = []

    for point_a, point_b, point_c in angle_triplets:
        angle = calculate_angle(
            coordinates[point_a],
            coordinates[point_b],
            coordinates[point_c],
        )

        # Converting degrees from roughly 0–180 into roughly 0–1. This keeps their numerical scale closer to my normalized coordinates.
        angles.append(angle / 180.0)

    return np.array(
        angles,
        dtype=np.float64,
    )


def extract_features(landmarks) -> np.ndarray:
    # Converting MediaPipe landmarks into the final ML feature vector.

    coordinates = landmarks_to_array(landmarks) # Converting MediaPipe objects into numerical coordinates.
    centered = center_landmarks(coordinates)    # Removing absolute screen position.
    normalized = normalize_scale(centered)  # Remove most of the effect of hand size/camera distance.

    # Flatten: (21, 3) into (63,)
    normalized_coordinates = normalized.flatten()

    # Angles describing how bent each finger is.
    joint_angles = extract_joint_angles(
        normalized
    )

    # Combining coordinate geometry and angle geometry.
    features = np.concatenate(
        [
            normalized_coordinates,
            joint_angles,
        ]
    )

    return features
    
    """
    Normalized coordinates retain:
    finger spacing
    relative fingertip positions
    palm geometry
    thumb location

    Angles retain:
    finger bending
    joint configuration
    """