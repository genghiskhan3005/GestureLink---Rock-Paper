"""
Gesture engine for GestureLink.

Converting stable ML pose predictions
into meaningful actions.

Gesture sequence:

FIST
  +
OPEN_PALM
  ↓
PASTE / TRANSFER EVENT
"""


import time



class GestureEngine:
    """
    Managing gesture sequences and
    converting them into actions.
    """


    def __init__(
        self,
        sequence_timeout: float = 2.0,
    ):
        """
        Initializing gesture memory.

        sequence_timeout:
        Maximum allowed time between
        two gesture steps.
        """


        # Storing the previous stable gesture.
        self.previous_gesture = None


        # Storing when the previous gesture
        # was detected.
        self.previous_time = None


        # Setting maximum time allowed
        # between gesture steps.
        self.sequence_timeout = (
            sequence_timeout
        )



    def update(
        self,
        gesture: str,
    ):
        """
        Receiving stable gestures and
        detecting gesture commands.
        """


        # Recording current time.
        current_time = time.time()



        # Checking whether the previous gesture
        # has expired.
        if (
            self.previous_time
            and
            current_time - self.previous_time
            > self.sequence_timeout
        ):

            # Clearing old gesture memory.
            self.previous_gesture = None



        action = None



        # Detecting the main GestureLink command:

        #
        # User 1:
        #
        # FIST
        #
        # (holding image)
        #
        # User 2:
        #
        # OPEN PALM
        #
        # Receiving image
        #
        #


        # Detecting palm -> fist sequence.
        
        # Meaning:
        # User is closing their hand.

        # Action:
        # COPY


        if (
            self.previous_gesture
            == "open_palm"
            and
            gesture
            == "fist"
        ):

            action = "copy"

            # Resetting stored gesture after
            # completing the copy action.
            self.previous_gesture = None



            # Detecting fist -> palm sequence.
            #
            # Meaning:
            # User is opening their hand.
            #
            # Action:
            # PASTE


        elif (
            self.previous_gesture
            == "fist"
            and
            gesture
            == "open_palm"
        ):

            action = "paste"

            # Resetting stored gesture after
            # completing the paste action.
            self.previous_gesture = None



        else:

            # Remembering current gesture
            # for future sequence detection.
            self.previous_gesture = gesture



        # Updating gesture timestamp.
        self.previous_time = current_time



        return action


    def reset(self):
        """
        Resetting stored gesture state.

        Clearing previous gesture memory so that
        a new gesture sequence can start cleanly.
        """


        # Removing previous gesture memory.
        self.previous_gesture = None


        # Removing previous gesture timestamp.
        self.previous_time = None
    