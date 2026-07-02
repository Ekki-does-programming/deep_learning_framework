# Simple training script for LeNet on MNIST

import os.path
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

import numpy as np
from Layers import Helpers
from Models.LeNet import build
import NeuralNetwork
import matplotlib.pyplot as plt

# Dataset
batch_size = 50
mnist = Helpers.MNISTData(batch_size)
# mnist.show_random_training_image()

# Build or load the LeNet model
SAVE_PATH = os.path.join(SCRIPT_DIR, 'trained', 'LeNet')
if os.path.isfile(SAVE_PATH):
    net = NeuralNetwork.load(SAVE_PATH, mnist)
else:
    net = build()
    net.data_layer = mnist

net.train(300)

NeuralNetwork.save(SAVE_PATH, net)

# Plot the loss
plt.figure('Loss function for training LeNet on the MNIST dataset')
plt.plot(net.loss, '-x', label='Training Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Validation
data, labels = net.data_layer.get_test_set()
results = net.test(data)

accuracy = Helpers.calculate_accuracy(results, labels)
print('On the MNIST test dataset, we achieve an accuracy of: ' + str(np.round(accuracy * 100, 4)) + '%')