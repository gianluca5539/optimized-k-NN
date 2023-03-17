# Description: Test the KNN algorithm.

import time
import numpy as np
import logging
import cv2

logging.basicConfig(level=logging.INFO)


def knn(
    test_image: np.ndarray,
    zipped_images_labels,
    k: int,
) -> int | np.int64:
    """K-Nearest Neighbors algorithm.

    Parameters
    ----------
    test_image : np.ndarray
        Array of the test image.
    zipped_images_labels : list[np.ndarray]
        List of tuples containing the training images and labels.
    k : int
        Number of neighbors to consider.

    Returns
    -------
    int
        The predicted label.
    """
    distances_list = np.array(
        [
            (np.linalg.norm(test_image - train_image), label)
            for train_image, label in zipped_images_labels
        ]
    )
    sorted_indexes = distances_list[:, 0].argsort()
    distances_list_sorted = distances_list[sorted_indexes]
    first_k_labels = distances_list_sorted[:k, 1].astype(np.int64)
    return np.bincount(first_k_labels).argmax()



def test_kNN(
    train_size: int, test_size: int, k: int, show_samples: bool = False
) -> float:
    """Test the KNN algorithm.

    Parameters
    ----------
    train_size : int
        The number of training images to use.
    test_size : int
        The number of images to predict.
    k : int
        The number of neighbors to consider.
    show_samples : bool, optional
        Whether to show the classified image with `cv2`, by default False

    Returns
    -------
    float
        The accuracy of the algorithm.
    """
    logging.info("Loading arrays...")
    images_train: np.ndarray = np.load("datasets/images_train.npy")[
        :train_size
    ]
    labels_train: np.ndarray = np.load("datasets/labels_train.npy")[
        :train_size
    ]
    images_test: np.ndarray = np.load("datasets/images_test.npy")[:test_size]
    labels_test: np.ndarray = np.load("datasets/labels_test.npy")[:test_size]

    zipped_images_labels = list(zip(images_train, labels_train))

    correct = 0

    logging.info("Running Tests...")
    for test_image, label in zip(images_test, labels_test):
        prediction = knn(test_image, zipped_images_labels, k)

        # show image, prediction and label
        if show_samples:
            logging.info(f"Prediction: {prediction}, Label: {label}")
            cv2.imshow("image", test_image.reshape(28, 28))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if prediction == label:
            correct += 1
    return correct / test_size


if __name__ == "__main__":
    logging.info("Testing the KNN algorithm...")
    start = time.time()
    accuracy = test_kNN(train_size=60000, test_size=50, k=3)
    end = time.time()
    logging.info(f"Time: {end - start:.2f} seconds")
    logging.info(f"The accuracy is: {accuracy:.2f}")
