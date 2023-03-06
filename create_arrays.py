"""
    This script creates numpy arrays from the MNIST dataset,
    extracting the data from the binary files into
    a more usable format.
"""

import numpy as np
import logging

logging.basicConfig(level=logging.INFO)


def create_images_array(path: str, test: bool):
    """Create a numpy array from the MNIST dataset.

    Result is saved as a .npy file.
    Parameters
    ----------
    path : str
        Path of the binary file.
    test : bool
        Whether the file is a test file or not.
    """

    with open(path, "rb") as f:
        _ = f.read(4)  # magic number
        num_images: int = int.from_bytes(f.read(4), "big")
        num_rows: int = int.from_bytes(f.read(4), "big")
        num_cols: int = int.from_bytes(f.read(4), "big")

        buf = f.read(num_rows * num_cols * num_images)

        array = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        array = array.reshape(num_images, num_rows * num_cols)
        logging.info(f"image array shape: {array.shape}")
        np.save(f'datasets/images_{"test" if test else "train"}.npy', array)


def create_labels_array(path: str, test: bool):
    """Load the labels from the MNIST dataset.

    Loads the labels from the binary file and saves them as a .npy file.
    Parameters
    ----------
    path : str
        Path of the binary file.
    test : bool
        Whether the file is a test file or not.
    """
    with open(path, "rb") as f:
        _ = f.read(4)  # magic number
        num_labels = int.from_bytes(f.read(4), "big")

        buf = f.read(num_labels)
        array = np.frombuffer(buf, dtype=np.uint8)

        logging.info(f"label array shape: {array.shape}")
        np.save(f'datasets/labels_{"test" if test else "train"}.npy', array)


def display_contents(arr_path: str, num_images: int = 10):
    """Display the contents of a numpy array.

    Uses the OpenCV library to display the images in windows,
    to install it run `pip install opencv-python`.

    Parameters
    ----------
    arr_path : str
        Path of the numpy array.
    num_images : int, optional
        Number of images to display, by default 10
    """
    import cv2

    arr = np.load(arr_path)
    # make it 2D
    arr = arr.reshape(arr.shape[0], 28, 28)
    for i in range(num_images):
        cv2.imshow(f"image {i}", arr[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.info("Creating arrays...")
    logging.info("creating train images array...")
    create_images_array("datasets/train-images.idx3-ubyte", test=False)
    logging.info("creating train labels array...")
    create_labels_array("datasets/train-labels.idx1-ubyte", test=False)
    logging.info("creating test images array...")
    create_images_array("datasets/t10k-images.idx3-ubyte", test=True)
    logging.info("creating test labels array...")
    create_labels_array("datasets/t10k-labels.idx1-ubyte", test=True)
    logging.info("Done!")

    logging.info("Displaying contents...")
    display_contents("datasets/images_train.npy", 3)
    display_contents("datasets/images_test.npy", 3)
