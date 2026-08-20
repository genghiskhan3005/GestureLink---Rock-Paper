"""
Action handler for GestureLink.

Receiving gesture events from the gesture engine
and executing the correct computer action.

Current supported actions:

copy
paste
"""


from pathlib import Path
import shutil



class ActionHandler:
    """
    Managing actions triggered by gestures.
    """


    def __init__(self):
        """
        Initializing stored transfer data.
        """


        # Storing the copied file path.
        self.copied_file = None



    def execute(
        self,
        action: str,
    ):
        """
        Executing actions received from
        the gesture engine.
        """


        # Checking whether a copy gesture
        # has been detected.
        if action == "copy":

            return self.copy()


        # Checking whether a paste gesture
        # has been detected.
        elif action == "paste":

            return self.paste()


        # Ignoring unknown actions.
        else:

            print(
                f"Unknown action: {action}"
            )

            return False



    def copy(
        self,
    ):
        """
        Handling copy event.

        Later this will:
            - capture selected image
            - store metadata
            - prepare transfer package
        """


        print(
            "COPY gesture detected."
        )


        # Temporary placeholder.
        # Real file selection comes later.
        self.copied_file = (
            "temporary_image.png"
        )


        return True



    def paste(
        self,
    ):
        """
        Handling paste event.

        Later this will:
            - receive transferred image
            - save it locally
            - display it
        """


        print(
            "PASTE gesture detected."
        )


        if self.copied_file:

            print(
                f"Pasting: {self.copied_file}"
            )

            return True


        print(
            "No copied item available."
        )


        return False