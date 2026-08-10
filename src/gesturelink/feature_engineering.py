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