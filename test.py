import numpy as np

@profile
def knn(test_image, zipped_images_labels, k):
    distances_list = [(np.linalg.norm(test_image - train_image), label) 
                      for train_image, label in zipped_images_labels]
    distances_list.sort(key=lambda x: x[0])
    k_nearest_neighbors = [label for _, label in distances_list[:k]]
    return max(k_nearest_neighbors, key=k_nearest_neighbors.count)

def test_kNN(train_size,test_size,k):
    images_train = np.load('datasets/images_train.npy')[:train_size]
    labels_train = np.load('datasets/labels_train.npy')[:train_size]
    images_test = np.load('datasets/images_test.npy')[:test_size]
    labels_test = np.load('datasets/labels_test.npy')[:test_size]

    zipped_images_labels = list(zip(images_train, labels_train))

    correct = 0
    for test_image, label in zip(images_test, labels_test):
        prediction = knn(test_image, zipped_images_labels, k)
        if prediction == label:
            correct += 1
    return correct/test_size


if __name__ == "__main__":
    accuracy = test_kNN(train_size=60000,test_size=100,k=3)
    print(accuracy)