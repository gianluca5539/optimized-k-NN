from load_dataset import create_images_array, create_labels_array

if __name__ == "__main__":
    images_train = create_images_array('datasets/train-images.idx3-ubyte', test=False)
    labels_train = create_images_array('datasets/train-labels.idx1-ubyte', test=False)
    images_test = create_labels_array('datasets/t10k-images.idx3-ubyte', test=True)
    labels_test = create_labels_array('datasets/t10k-labels.idx1-ubyte', test=True)

    print(images_train.shape)
