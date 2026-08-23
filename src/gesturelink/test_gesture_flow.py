"""
Testing the complete GestureLink gesture logic
without using the webcam.

Simulating:

OPEN PALM → FIST
        ↓
       COPY


FIST → OPEN PALM
        ↓
       PASTE
"""


from gesture_engine import GestureEngine
from action_handler import ActionHandler



def simulate_gesture_sequence(
    gestures,
    gesture_engine,
    action_handler,
):
    """
    Sending simulated gestures through
    the gesture engine and action handler.
    """

    # Sending each simulated gesture.
    for gesture in gestures:


        print(
            f"\nGesture detected: {gesture}"
        )


        # Asking gesture engine whether
        # a meaningful action happened.
        action = (
            gesture_engine
            .update(gesture)
        )


        # Executing action if one exists.
        if action:


            print(
                f"Action triggered: {action}"
            )


            action_handler.execute(
                action
            )


        else:

            print(
                "No action triggered."
            )



if __name__ == "__main__":


    # Creating one shared gesture engine.
    gesture_engine = GestureEngine()


    # Creating one shared action handler.
    action_handler = ActionHandler()

    action_handler.set_selected_file(
    "tests/shawshank_redemption.jpg"
)



    print("=" * 50)
    print("TEST 1: COPY GESTURE")
    print("PALM → FIST")
    print("=" * 50)


    simulate_gesture_sequence(
        [
            "open_palm",
            "fist",
        ],
        gesture_engine,
        action_handler,
    )



    print("\n\n")



    print("=" * 50)
    print("TEST 2: PASTE GESTURE")
    print("FIST → PALM")
    print("=" * 50)

    gesture_engine.reset()

    simulate_gesture_sequence(
        [
            "fist",
            "open_palm",
        ],
        gesture_engine,
        action_handler,
    )