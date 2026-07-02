import numpy as np
from Layers import Conv, FullyConnected, SoftMax, ReLU, Flatten, Pooling
from Optimization.Optimizers import Adam
from Optimization.Constraints import L2_Regularizer
from Layers.Initializers import He
from NeuralNetwork import NeuralNetwork
from Optimization.Loss import CrossEntropyLoss

# LeNet
def build():
    # configure optimizer
    optimizer = Adam(learning_rate=5e-4, mu=0.9, rho=0.9)
    optimizer.add_regularizer(L2_Regularizer(4e-4))
    # build the model
    model = NeuralNetwork(optimizer, He(), He())
    # build the stated layers with their respective parameters
    # see slide 14 (or wikipedia :D [keeping in mind slide 15])
    model.append_layer(Conv.Conv(stride_shape=(1, 1), convolution_shape=(1, 5, 5), num_kernels=6))
    model.append_layer(ReLU.ReLU())
    model.append_layer(Pooling.Pooling(pooling_shape=(2, 2), stride_shape=(2, 2)))
    model.append_layer(Conv.Conv(stride_shape=(1, 1), convolution_shape=(6, 5, 5), num_kernels=16))
    model.append_layer(ReLU.ReLU())
    model.append_layer(Pooling.Pooling(pooling_shape=(2, 2), stride_shape=(2, 2)))
    model.append_layer(Flatten.Flatten())
    model.append_layer(FullyConnected.FullyConnected(input_size=16*7*7, output_size=120))
    model.append_layer(ReLU.ReLU())
    model.append_layer(FullyConnected.FullyConnected(input_size=120, output_size=84))
    model.append_layer(ReLU.ReLU())
    model.append_layer(FullyConnected.FullyConnected(input_size=84, output_size=10))
    model.append_layer(SoftMax.SoftMax())
    # set the loss layer for loss computation
    model.loss_layer = CrossEntropyLoss()
    return model