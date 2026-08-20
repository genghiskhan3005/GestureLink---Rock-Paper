"""
Prediction smoothing module for GestureLink.

Reducing frame-to-frame prediction noise while
keeping gesture transitions responsive.
"""


from collections import deque
from collections import Counter


class PredictionSmoother:
    """
    Smoothing ML predictions using a short
    rolling prediction history.
    """


    def __init__(
        self,
        window_size: int = 5,
    ):
        """
        Initializing prediction history.

        Smaller windows:
            Faster response
            Less smoothing

        Larger windows:
            Slower response
            More stability
        """


        # Creating a fixed-size queue for
        # storing recent predictions.
        self.prediction_history = deque(
            maxlen=window_size
        )


    def update(
        self,
        prediction: str,
    ) -> str:
        """
        Adding a new prediction and returning
        the most stable current gesture.
        """


        # Adding the newest prediction
        # into the history buffer.
        self.prediction_history.append(
            prediction
        )


        # Counting occurrences of each
        # gesture in recent frames.
        counts = Counter(
            self.prediction_history
        )


        # Returning the gesture appearing
        # most frequently.
        stable_prediction = (
            counts
            .most_common(1)[0][0]
        )


        return stable_prediction



    def reset(self) -> None:
        """
        Clearing previous predictions when
        tracking is interrupted.
        """


        # Removing stored predictions.
        self.prediction_history.clear()