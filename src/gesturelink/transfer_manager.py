"""
Transfer manager for GestureLink.

Managing copied files before sending them
to another device.

Current stage:
Local transfer simulation

Future:
Laptop ↔ Laptop
Laptop ↔ Phone
"""


from pathlib import Path
import shutil



class TransferManager:
    """
    Managing file copy and retrieval.
    """


    def __init__(
        self,
        storage_folder="transfer_buffer",
    ):
        """
        Initializing transfer storage.
        """


        # Creating a temporary storage location
        # for copied files.
        self.storage_folder = Path(
            storage_folder
        )


        # Creating the folder if it does not exist.
        self.storage_folder.mkdir(
            exist_ok=True
        )


        # Storing currently copied file.
        self.current_file = None



    def copy_file(
        self,
        file_path: str,
    ):
        """
        Copying a file into the transfer buffer.
        """


        # Converting string path into Path object.
        source = Path(
            file_path
        )


        # Checking whether the file exists.
        if not source.exists():

            print(
                "File does not exist."
            )

            return False



        # Creating destination path.
        destination = (
            self.storage_folder
            /
            source.name
        )


        # Copying the file.
        shutil.copy2(
            source,
            destination,
        )


        # Saving copied file information.
        self.current_file = destination


        print(
            f"Copied: {destination.name}"
        )


        return True



    def paste_file(
        self,
        destination_folder: str,
    ):
        """
        Pasting the copied file into
        a destination folder.
        """


        # Checking whether a file
        # has been copied.
        if self.current_file is None:

            print(
                "Nothing to paste."
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
            self.current_file.name
        )


        # Restoring the copied file.
        shutil.copy2(
            self.current_file,
            destination,
        )


        print(
            f"Pasted: {destination}"
        )


        return True