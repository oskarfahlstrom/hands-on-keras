import cv2
import numpy as np
from matplotlib import colors
import tensorflow as tf


def get_bgr_color(name):
    """Convert a named color to BGR format."""
    return tuple(int(c * 255) for c in reversed(colors.to_rgb(name)))

def adjust_bgr_color(color, depth_shade):
    """Adjust each channel and clamp between 0 and 255."""
    return tuple(max(0, min(255, c - depth_shade)) for c in color)


# some constants
IMAGE_DIMENSIONS = (150, 150)  # to match keras model dimensions
LABELS = ["ROCK", "PAPER", "SCISSORS"]


def run(cam_feed: int = 0):
    """Launch a hand recognition session with webcam feed."""
    # initialize webcam
    model = tf.keras.models.load_model("resources/model.h5", compile=False)
    cam = cv2.VideoCapture(cam_feed)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 500)

    while True:
        success, img = cam.read()
        if not success:
            break

        # flip image for natural mirroring and convert to RGB
        img_rgb = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)

        # preprocess frame for the model
        input_frame = cv2.resize(img_rgb, IMAGE_DIMENSIONS)
        input_frame = input_frame / 255.0  # normalize to [0, 1]
        input_frame = np.expand_dims(input_frame, axis=0)

        # apply model and make predictions
        predictions = model.predict(input_frame)
        gesture_index = np.argmax(predictions)  # get the index of the highest confidence score
        recognized_gesture = LABELS[gesture_index]  # map index to gesture label
        print(recognized_gesture)

        cv2.imshow("Hand Gesture Recognition", img)

        if cv2.waitKey(1) != -1:  # exit on any key press
            break

    # cleanup
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run(0)  # webcam feed 0
