"""
    This script creates numpy arrays from the MNIST dataset,
    extracting the data from the binary files into
    a more usable format.
"""

import numpy as np
import logging

logging.basicConfig(level=logging.INFO)


def create_images_matrix(f: list) -> np.ndarray:
    """Create a matrix from the MNIST file.

    Parameters
    ----------
    f : list > The file to read from.
        
    Returns
    -------
    np.ndarray > The matrix.
    """
    _ = f.read(4)  # magic number
    num_images: int = int.from_bytes(f.read(4), "big")
    num_rows: int = int.from_bytes(f.read(4), "big")
    num_cols: int = int.from_bytes(f.read(4), "big")
    buf = f.read(num_rows * num_cols * num_images)
    array = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    images_matrix = array.reshape(num_images, num_rows * num_cols)
    return images_matrix


def create_images_arrays(paths: str, hyperparam: int) -> None:
    """Create numpy arrays from the MNIST dataset.

    This time the matrices are transformed using PCA.
    Result is saved as a .npy file.

    Parameters
    ----------
    paths : tuple[str, str]
        Paths of the training and test image sets.
    """
    train_pc_matrix = None # this will be used to transform the test set
    # Train set
    with open(paths[0], "rb") as f:
        images_matrix = create_images_matrix(f)
        logging.info(f"image array shape: {images_matrix.shape}")

        # first we need to center the data around 0,0 and scale it
        images_matrix_std = images_matrix.copy()
        mean = np.mean(images_matrix, axis=0) # mean of each column
        images_matrix_std -= mean # subtract mean from each column to center the data around 0,0
        images_matrix_std /= np.std(images_matrix, axis=0) # divide by standard deviation
        # now we have a matrix with mean 0 and standard deviation 1, let's apply PCA
        cov = np.cov(images_matrix, rowvar=False) # covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(cov) # eigenvalues and eigenvectors of the covariance matrix
        eigenvalues_indexes_descending = eigenvalues.argsort()[::-1] # indexes of the eigenvalues in descending order
        eigenvectors_descending = eigenvectors[:, eigenvalues_indexes_descending][:hyperparam] # eigenvectors in descending order, limited to hyperparameter
        pcmatrix_computed = eigenvectors_descending.T # turn the vectors into row vectors
        images_pca = images_matrix @ pcmatrix_computed # this gives us the matrix with reduced dimensionality
        train_pc_matrix = pcmatrix_computed # store the matrix for later use in the test set

        # save the matrix
        logging.info(f"reduced array shape: {images_pca.shape}")
        np.save(f'datasets/images_train.npy', images_pca)
    
    # Test set
    with open(paths[1], "rb") as f:
        images_matrix = create_images_matrix(f)
        logging.info(f"image array shape: {images_matrix.shape}")
        images_pca = images_matrix @ train_pc_matrix # this gives us the matrix with reduced dimensionality
        logging.info(f"reduced array shape: {images_pca.shape}")
        np.save(f'datasets/images_test.npy', images_pca)


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

if __name__ == "__main__":
    logging.info("Creating arrays...")
    logging.info("creating train and test images array...")
    create_images_arrays(("datasets/train-images.idx3-ubyte","datasets/t10k-images.idx3-ubyte"), hyperparam=222)
    logging.info("creating train labels array...")
    create_labels_array("datasets/train-labels.idx1-ubyte", test=False)
    logging.info("creating test labels array...")
    create_labels_array("datasets/t10k-labels.idx1-ubyte", test=True)
    logging.info("Done!")

