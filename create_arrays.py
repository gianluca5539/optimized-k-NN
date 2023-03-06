import numpy as np
import logging

logging.basicConfig(level=logging.INFO)


def create_images_array(path: str, test):
    with open(path, "rb") as f:
        _ = f.read(4)  # magic number
        num_images = int.from_bytes(f.read(4), "big")
        num_rows = int.from_bytes(f.read(4), "big")
        num_cols = int.from_bytes(f.read(4), "big")
        array = np.array(
            [
                [
                    int.from_bytes(f.read(1), "big")
                    for _ in range(num_cols)
                    for _ in range(num_rows)
                ]
                for _ in range(num_images)
            ],
            dtype=np.uint8,
        )
        np.save(f'datasets/images_{"test" if test else "train"}.npy', array)


def create_labels_array(path: str, test):
    with open(path, "rb") as f:
        _ = f.read(4)  # magic number
        num_labels = int.from_bytes(f.read(4), "big")
        array = np.array(
            [int.from_bytes(f.read(1), "big") for _ in range(num_labels)]
        )
        np.save(f'datasets/labels_{"test" if test else "train"}.npy', array)


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
