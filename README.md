# optimized k-NN

In this repo, I'm making the fastest implementation of the k-Nearest Neighbors algorithm I can manage to code.

K-Nearest Neighbors (KNN) is a supervised machine learning algorithm used for classification and regression tasks. It makes predictions based on the similarity of a new data point to the training data.

# Key Idea
The MNIST has made available a set of 60.000 28x28 images representing handwritten numbers. This algorithm performs OCR, recognizing an handwritten number it has never seen before.
Flattening the 28x28 matrix representing an image, we get a 784-item vector, that we can see as a point in 784-dimensional space.
The algorithm computes the Euclidean distance between the point given by the test image and each point of the 60.000 images of the training set. Then, it returns the most frequent label among the k nearest points.



# Libraries I used
NumPy has been used to make things faster since it's coded in C.

# How to use it
To test this repo, you can easily clone it and run test.py. Be sure to build the .npy files first by running create_arrays.py.

# NumPy files
Four .npy files need to be generated in order to run the algorighm. This process should not take very long thanks to @dario-loi's optimization using buffers.
