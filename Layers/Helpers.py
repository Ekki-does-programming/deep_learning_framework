import numpy as np
import matplotlib.pyplot as plt
import os
import gzip
import struct
import random
from pathlib import Path
from random import shuffle

def compute_bn_gradients(error_tensor, input_tensor, weights, mean, var, eps=np.finfo(float).eps):
    # computation of the gradient w.r.t the input for the batch_normalization layer

    if eps > 1e-10:
        raise ArithmeticError("Eps must be lower than 1e-10. Your eps values %s" %(str(eps)))

    norm_mean = input_tensor - mean
    var_eps = var + eps

    gamma_err = error_tensor * weights
    inv_batch = 1. / error_tensor.shape[0]

    grad_var = np.sum(norm_mean * gamma_err * -0.5 * (var_eps ** (-3 / 2)), keepdims=True, axis=0)

    sqrt_var = np.sqrt(var_eps)
    first = gamma_err * 1. / sqrt_var

    grad_mu_two = (grad_var * np.sum(-2. * norm_mean, keepdims=True, axis=0)) * inv_batch
    grad_mu_one = np.sum(gamma_err * -1. / sqrt_var, keepdims=True, axis=0)

    second = grad_var * (2. * norm_mean) * inv_batch
    grad_mu = grad_mu_two + grad_mu_one

    return first + second + inv_batch * grad_mu


def calculate_accuracy(results, labels):

    index_maximum = np.argmax(results, axis=1)
    one_hot_vector = np.zeros_like(results)
    for i in range(one_hot_vector.shape[0]):
        one_hot_vector[i, index_maximum[i]] = 1

    correct = 0.
    wrong = 0.
    for column_results, column_labels in zip(one_hot_vector, labels):
        if column_results[column_labels > 0.].all() > 0.:
            correct += 1.
        else:
            wrong += 1.

    return correct / (correct + wrong)


def shuffle_data(input_tensor, label_tensor):
    index_shuffling = [i for i in range(input_tensor.shape[0])]
    shuffle(index_shuffling)
    shuffled_input = [input_tensor[i, :] for i in index_shuffling]
    shuffled_labels = [label_tensor[i, :] for i in index_shuffling]
    return (np.array(shuffled_input)), (np.array(shuffled_labels))

class MNISTData:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.train, self.labels = self._read()
        self.test, self.testLabels = self._read(dataset="testing")

        self._current_forward_idx_iterator = self._forward_idx_iterator()

    def _forward_idx_iterator(self):
        num_iterations = int(self.train.shape[0] / self.batch_size)
        idx = np.arange(self.train.shape[0])
        while True:
            this_idx = np.random.choice(idx, self.train.shape[0], replace=False)
            for i in range(num_iterations):
                yield this_idx[i * self.batch_size:(i + 1) * self.batch_size]

    def next(self):
        idx = next(self._current_forward_idx_iterator)
        return self.train[idx, :], self.labels[idx, :]

    def show_random_training_image(self):
        image = self.train[np.random.randint(0, self.train.shape[0]-1), :28 , :28]
        plt.imshow(image.reshape(28, 28), cmap='gray')
        plt.show()

    def show_image(self, index, test=True):
        if test:
            image = self.test[index, :28 * 28]
        else:
            image = self.train[index, :28 * 28]

        plt.imshow(image.reshape(28, 28), cmap='gray')
        plt.show()

    def get_test_set(self):
        return self.test, self.testLabels

    @staticmethod
    def _read(dataset="training"):
        """
        Python function for importing the MNIST data set.  It returns an iterator
        of 2-tuples with the first element being the label and the second element
        being a numpy.uint8 2D array of pixel data for the given image.
        """

        root_dir = Path(__file__)

        if dataset == "training":
            fname_img = root_dir.parent.parent.joinpath('MNIST', 'Data', 'train-images-idx3-ubyte.gz')
            fname_lbl = root_dir.parent.parent.joinpath('MNIST', 'Data', 'train-labels-idx1-ubyte.gz')
        elif dataset == "testing":
            fname_img = root_dir.parent.parent.joinpath('MNIST', 'Data', 't10k-images-idx3-ubyte.gz')
            fname_lbl = root_dir.parent.parent.joinpath('MNIST', 'Data', 't10k-labels-idx1-ubyte.gz')
        else:
            raise ValueError("dataset must be 'testing' or 'training'")

        # Load everything in some numpy arrays
        with gzip.open(str(fname_lbl), 'rb') as flbl:
            magic, num = struct.unpack(">II", flbl.read(8))

            s = flbl.read(num)
            lbl = np.frombuffer(s, dtype=np.int8)
            one_hot = np.zeros((lbl.shape[0],10))
            for idx, l in enumerate(lbl):
                one_hot[idx, l] = 1

        with gzip.open(str(fname_img), 'rb') as fimg:
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))

            buffer = fimg.read(num * 32 * 32 * 8)
            img = np.frombuffer(buffer, dtype=np.uint8).reshape(len(lbl), 1, rows,  cols)
            img = img.astype(np.float64)
            img /= 255.0

        img = img[:num, :]
        one_hot = one_hot[:num, :]
        return img, one_hot
