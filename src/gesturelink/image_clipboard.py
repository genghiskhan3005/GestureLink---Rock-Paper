"""
Image clipboard system for GestureLink.

Managing copied images before transferring
them between devices.

Current responsibility:

COPY:
    Store image information

PASTE:
    Retrieve stored image

Future:
    Replace local storage with network transfer
"""


from pathlib import Path
import shutil



class ImageClipboard:
    """
    Managing GestureLink's image clipboard.
    """


    def __init__(
        self,
        clipboard_folder="image_clipboard",
    ):
        """
        Initializing clipboard storage.
        """


        # Creating local clipboard folder.
        self.clipboard_folder = Path(
            clipboard_folder
        )


        self.clipboard_folder.mkdir(
            exist_ok=True
        )


        # Storing currently copied image.
        self.current_image = None



    def copy_image(
        self,
        image_path: str,
    ):
        """
        Copying an image into GestureLink clipboard.
        """


        # Converting string path into Path object.
        source = Path(
            image_path
        )


        # Checking whether image exists.
        if not source.exists():

            print(
                "Image does not exist."
            )

            return False



        # Creating clipboard destination.
        destination = (
            self.clipboard_folder
            /
            source.name
        )


        # Copying image into clipboard.
        shutil.copy2(
            source,
            destination,
        )


        # Remembering copied image.
        self.current_image = destination


        print(
            f"Image copied: {destination.name}"
        )


        return True



    def paste_image(
        self,
        destination_folder="pasted_images",
    ):
        """
        Pasting the copied image.
        """


        # Checking whether clipboard
        # contains an image.
        if self.current_image is None:

            print(
                "Clipboard is empty."
            )

            return False



        destination_folder = Path(
            destination_folder
        )


        destination_folder.mkdir(
            exist_ok=True
        )


        destination = (
            destination_folder
            /
            self.current_image.name
        )


        # Restoring the image.
        shutil.copy2(
            self.current_image,
            destination,
        )


        print(
            f"Image pasted: {destination}"
        )


        return True