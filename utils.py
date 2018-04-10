from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import pickle

# Model construction utilities below adapted from
# https://www.tensorflow.org/versions/r0.8/tutorials/mnist/pros/index.html#deep-mnist-for-experts
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def shuffle_aligned_list(data):
    """Shuffle arrays in a list by shuffling each array identically."""
    num = data[0].shape[0]
    p = np.random.permutation(num)
    return [d[p] for d in data]


def batch_generator(data, batch_size, shuffle=True):
    """Generate batches of data.
    
    Given a list of array-like objects, generate batches of a given
    size by yielding a list of array-like objects corresponding to the
    same slice of each input.
    """
    if shuffle:
        data = shuffle_aligned_list(data)

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= len(data[0]):
            batch_count = 0

            if shuffle:
                data = shuffle_aligned_list(data)

        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data]


def imshow_grid(images, shape=[2, 8]):
    """Plot images in a grid of a given shape."""
    fig = plt.figure(1)
    grid = ImageGrid(fig, 111, nrows_ncols=shape, axes_pad=0.05)

    size = shape[0] * shape[1]
    for i in range(size):
        grid[i].axis('off')
        grid[i].imshow(images[i])  # The AxesGrid object work as a list of axes.

    plt.show()


def plot_embedding(X, y, d, title=None):
    """Plot an embedding X with the class label y colored by the domain d."""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    plt.figure(figsize=(10,10))
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        # plot colored number
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.bwr(d[i] / 1.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

def load_mnist_M(dataset_name):
    """Dataset class for MNIST-M.
    Args:
        split ({'train', 'valid', 'test'}): Select a split of the dataset.
        withlabel (bool): If ``True``, dataset returns a tuple of an image and
            a label. Otherwise, the datasets only return an image.
        scale (float): Pixel value scale. If it is 1 (default), pixels are
            scaled to the interval ``[0, 1]``.
        dtype: Data type of resulting image arrays.
        label_dtype: Data type of the labels.
    """
    data_dir = os.path.join("./data/mnistm", dataset_name + ".pkl")
    with open(data_dir, 'rb') as f:
        data = pickle.load(f)
        images_train = data['train']['images'].astype(np.float32)
        labels_train = data['train']['labels'].astype(np.int32)
        images_test = data['test']['images'].astype(np.float32)
        labels_test = data['test']['labels'].astype(np.int32)
        images_valid = data['valid']['images'].astype(np.float32)
        labels_valid = data['valid']['labels'].astype(np.int32)

    images_train *= (1 / 255.0)
    images_test *= (1 / 255.0)
    images_valid *= (1 / 255.0)

    images_train = np.transpose(images_train, axes=[0, 2, 3, 1])
    images_test = np.transpose(images_test, axes=[0, 2, 3, 1])
    images_valid = np.transpose(images_valid, axes=[0, 2, 3, 1])

    # seed = 547
    # np.random.seed(seed)  # 确保每次生成的随机数相同
    # np.random.shuffle(images_train)  # 将mnist数据集中数据的位置打乱
    # np.random.seed(seed)
    # np.random.shuffle(labels_train)

    y_vec = np.zeros((len(labels_train), 10), dtype=np.float)

    for i, label in enumerate(labels_train):
        y_vec[i, labels_train[i]] = 1.0

    return images_train, images_test, images_valid, y_vec