import numpy as np
import logging

logging.basicConfig(level=logging.INFO)


def knn(test_image: np.ndarray, zipped_images_labels, k: int) -> int:
    distances_list = np.array(
        [
            (np.linalg.norm(test_image - train_image), label)
            for train_image, label in zipped_images_labels
        ]
    )
    sorted_indexes = distances_list[:, 0].argsort()
    distances_list_sorted = distances_list[sorted_indexes]
    first_k_labels = distances_list_sorted[:k, 1].astype(int)
    return np.bincount(first_k_labels).argmax()


def test_kNN(train_size: int, test_size: int, k: int) -> float:
    logging.info("Loading arrays...")
    images_train = np.load("datasets/images_train.npy")[:train_size]
    labels_train = np.load("datasets/labels_train.npy")[:train_size]
    images_test = np.load("datasets/images_test.npy")[:test_size]
    labels_test = np.load("datasets/labels_test.npy")[:test_size]

    zipped_images_labels = list(zip(images_train, labels_train))

    correct = 0

    logging.info("Running Tests...")
    for test_image, label in zip(images_test, labels_test):
        prediction = knn(test_image, zipped_images_labels, k)
        if prediction == label:
            correct += 1
    return correct / test_size


if __name__ == "__main__":
    logging.info("Testing the KNN algorithm...")
    accuracy = test_kNN(train_size=60000, test_size=50, k=3)
    logging.info(f"The accuracy is: {accuracy:.2f}")
