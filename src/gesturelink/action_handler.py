"""
Action handler for GestureLink.

Receiving gesture events from the gesture engine
and connecting them to file transfer actions.
"""


from transfer_manager import TransferManager



class ActionHandler:
    """
    Managing gesture-triggered actions.
    """


    def __init__(self):
        """
        Initializing the action system.
        """


        # Creating the transfer manager.
        self.transfer_manager = TransferManager()



        # Storing a test image path.
        # Later this will come from the selected image.
        self.selected_file = None



    def set_selected_file(
        self,
        file_path: str,
    ):
        """
        Selecting the file that will be copied.
        """


        # Saving the selected file path.
        self.selected_file = file_path



    def execute(
        self,
        action: str,
    ):
        """
        Executing actions received from
        the gesture engine.
        """


        if action == "copy":

            return self.copy()



        elif action == "paste":

            return self.paste()



        else:

            print(
                f"Unknown action: {action}"
            )

            return False



    def copy(self):
        """
        Handling copy gesture.
        """


        if self.selected_file is None:

            print(
                "No file selected for copying."
            )

            return False



        # Sending the selected file
        # into the transfer buffer.
        return self.transfer_manager.copy_file(
            self.selected_file
        )



    def paste(self):
        """
        Handling paste gesture.
        """


        # Creating a destination folder
        # for testing paste.
        return self.transfer_manager.paste_file(
            "pasted_files"
        )