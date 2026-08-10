# Rule-based hand pose recognition(fist/palm) for GestureLink.

import math


FINGER_STRAIGHT_ANGLE = 160.0   # We consider finger angle of minimum 160 to be straight fingers


def calculate_angle(point_a, point_b, point_c) -> float:
    # Calculates the angle ABC in degrees. point_b is the middle point where the angle is measured.
    # For a finger, we can measure MCP - PIP - DIP

    vector_ba = (   # Creates a vector going from B to A
        point_a.x - point_b.x,
        point_a.y - point_b.y,
        point_a.z - point_b.z,
    )

    vector_bc = (
        point_c.x - point_b.x,
        point_c.y - point_b.y,
        point_c.z - point_b.z,
    )
    # Now we have 2 vectors, one going from B to A, another going from B to C and B is in the middle, so the angle between these vectors is the angle we want

    dot_product = sum(
        a * c for a, c in zip(vector_ba, vector_bc) # zip pairs the axes -> (x1,x2), (y1,y2), (z1,z2) # a*c multiplies them and then they are summed -> Basically dot product x1x2 + y1y2 + z1z2
    )

    magnitude_ba = math.sqrt(
        sum(value * value for value in vector_ba)
    )

    magnitude_bc = math.sqrt(
        sum(value * value for value in vector_bc)
    )

    if magnitude_ba == 0 or magnitude_bc == 0:
        return 0.0

    cosine_angle = dot_product / (
        magnitude_ba * magnitude_bc
    )

    # Floating-point calculations can sometimes produce values
    # slightly outside the valid range for acos.
    cosine_angle = max(-1.0, min(1.0, cosine_angle))    # Ensures floating point error is solved

    return math.degrees(math.acos(cosine_angle))    # acos is basically inverse # converts the cosine value back into an angle


def is_finger_extended(
    landmarks,
    mcp_index: int,
    pip_index: int,
    dip_index: int,
) -> bool:
    # Returns True when a finger is approximately straight
    # For the index finger, MediaPipe numbers them: 5 = MCP, 6 = PIP, 7 = DIP, 8 = fingertip

    angle = calculate_angle(
        landmarks[mcp_index],
        landmarks[pip_index],
        landmarks[dip_index],
    )

    return angle >= FINGER_STRAIGHT_ANGLE


def count_extended_fingers(landmarks) -> int:
    # Counts extended index, middle, ring, and pinky fingers.
    # Thumb is not considered

    finger_landmarks = [
        (5, 6, 7),      # Index
        (9, 10, 11),    # Middle
        (13, 14, 15),   # Ring
        (17, 18, 19),   # Pinky
    ]   # Each tuple contains (MCP, PIP, DIP)

    extended_count = 0

    for mcp, pip, dip in finger_landmarks:
        if is_finger_extended(
            landmarks,
            mcp,
            pip,
            dip,
        ):
            extended_count += 1

    return extended_count


def classify_pose(landmarks) -> str:
    # Classify a hand as open palm, fist, or other.

    extended_fingers = count_extended_fingers(landmarks)

    if extended_fingers == 4:
        return "open_palm"  # 4 fingers being straight is considered an open palm

    if extended_fingers <= 1:
        return "fist"   # Less than 2 fingers being straight is considered a fist

    return "other" # 2 or 3 fingers being straight is considerend ambigious