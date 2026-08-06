"""Basic webcam test"""

import cv2


def run_webcam() -> None:   # -> None means this function returns no value 

    # Camera index 0 normally means the built-in/default webcam.
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError(
            "The webcam could not be opened. "
            "Check camera permissions or close other camera applications."
        )

    print("Webcam started.")
    print("Press E to close it.")

    try:
        while True:
            frame_received, frame = camera.read()   # frame is returned as a NumPy array of pixels

            if not frame_received:
                print("A webcam frame could not be read.")  # eg. webcame was suddenly unplugged
                break

        
            frame = cv2.flip(frame, 1)  # flips the image horizontally across the vertical axis(1)

            cv2.putText(
                frame,
                "GestureLink - Rock & Paper\nPress E to exit",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (30, 30, 30),    # color in BGR format (dark gray)
                2,  # line thickness in pixels
            )

            cv2.imshow("GestureLink - Rock & Paper Webcam", frame) # renders the modified frame inside a GUI window titled "GestureLink - Rock & Paper Webcam"
            
            pressed_key = cv2.waitKey(1) & 0xFF # .waitkey(1) pauses execution for 1 millisecond to display the current frame and wait for any keyboard input.  # the bitwise AND operation (& 0xFF) acts as a filter that strips away all upper bits and keeps only the last 8 bits (1 byte), which contains the true ASCII value, this makes the code suitable across different OSs otherwise .waitKey(1) might return return different values for linux and windows
            # in OpenCV, windows need time to process GUI events (like refreshing the displayed image). Without .waitKey(), the display window will freeze or crash.

            if pressed_key == ord("e"):
                break

    finally:    # a block that is guaranteed to run after try block finishes or if an error terminates the loop
        camera.release() # release the webcam even if an error occurs.
        cv2.destroyAllWindows() # closes all OpenCV GUI windows that were created during execution
        print("Webcam closed safely.")


if __name__ == "__main__":  # ensures that run_webcam() is called only when this script is run directly (not when imported as a module into another script)
    run_webcam()