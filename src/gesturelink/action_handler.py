"""
Action handler for GestureLink.

Connecting gesture commands with the
image clipboard system.

Supported actions:

copy
paste
"""


from image_clipboard import ImageClipboard



class ActionHandler:
    """
    Managing gesture-triggered image actions.
    """


    def __init__(self):
        """
        Initializing image clipboard.
        """


        # Creating the GestureLink image clipboard.
        self.image_clipboard = ImageClipboard()



        # Storing the currently selected image.
        self.selected_image = None



    def set_selected_image(
        self,
        image_path: str,
    ):
        """
        Selecting an image to copy.
        """


        # Saving the selected image path.
        self.selected_image = image_path



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
        Handling COPY gesture.

        Copying selected image into
        GestureLink clipboard.
        """


        if self.selected_image is None:

            print(
                "No image selected."
            )

            return False



        return self.image_clipboard.copy_image(
            self.selected_image
        )



    def paste(self):
        """
        Handling PASTE gesture.

        Restoring image from
        GestureLink clipboard.
        """


        return self.image_clipboard.paste_image()