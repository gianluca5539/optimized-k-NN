import numpy as np


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
    create_images_array("datasets/train-images.idx3-ubyte", test=False)
    create_labels_array("datasets/train-labels.idx1-ubyte", test=False)
    create_images_array("datasets/t10k-images.idx3-ubyte", test=True)
    create_labels_array("datasets/t10k-labels.idx1-ubyte", test=True)
