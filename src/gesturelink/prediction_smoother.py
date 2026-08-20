"""
Adaptive prediction smoothing for GestureLink.

Reducing ML prediction noise while allowing
fast gesture transitions.

Logic:

Current stable gesture
        |
        ↓
New prediction arrives
        |
        ↓
If it matches:
    keeping it immediately

If it is different:
    counting consecutive confirmations

After enough confirmations:
    switching gesture
"""


class PredictionSmoother:
    """
    Creating an adaptive gesture smoother.

    Keeping the current gesture stable while
    detecting intentional gesture changes.
    """


    def __init__(
        self,
        confirmation_frames: int = 3,
    ):
        """
        Initializing smoother settings.

        confirmation_frames:
        Number of consecutive new predictions
        required before accepting a change.
        """


        # Storing the currently accepted gesture.
        self.current_prediction = None


        # Storing the possible new gesture
        # that is waiting for confirmation.
        self.candidate_prediction = None


        # Counting how many times the candidate
        # gesture has appeared consecutively.
        self.candidate_count = 0


        # Setting how many frames are required
        # before switching gestures.
        self.confirmation_frames = (
            confirmation_frames
        )


    def update(
        self,
        prediction: str,
    ) -> str:
        """
        Updating the smoother with a new ML prediction.
        """


        # Initializing the first gesture immediately.
        if self.current_prediction is None:

            self.current_prediction = prediction

            return self.current_prediction



        # If prediction matches the current gesture,
        # keeping it and clearing any transition.
        if prediction == self.current_prediction:


            self.candidate_prediction = None

            self.candidate_count = 0


            return self.current_prediction



        # If prediction is different from current,
        # checking whether it is a possible transition.
        if prediction == self.candidate_prediction:


            # Increasing confirmation count.
            self.candidate_count += 1


        else:


            # Starting a new candidate gesture.
            self.candidate_prediction = prediction

            self.candidate_count = 1



        # Switching only after enough confirmations.
        if (
            self.candidate_count
            >= self.confirmation_frames
        ):


            self.current_prediction = (
                self.candidate_prediction
            )


            self.candidate_prediction = None

            self.candidate_count = 0



        return self.current_prediction



    def reset(self):
        """
        Resetting smoother state.
        """


        # Removing stored gesture history.
        self.current_prediction = None


        self.candidate_prediction = None


        self.candidate_count = 0