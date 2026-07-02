import numpy as np
from .Helpers import compute_bn_gradients

from .Base import BaseLayer
# Batch Normalization Regularization
class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.trainable = True
        # channel count
        self.channels = channels
        # weights and bias
        self.weights = None
        self.bias = None
        # gradients
        self.gradient_weights = None
        self.gradient_bias = None
        # moving mean and variance
        self.moving_mean = None
        self.moving_variance = None
        self.momentum = 0.8
        # optimizer
        self.optimizer = None
        # epsilon for numerical stability
        self.epsilon = 1e-10
        # store the tensor shape for reshaping
        self.tensor_shape = None
        # store tensor for backward pass
        self.input_tensor = None
        self.normalized_input = None
        # layer behaviour (4D to 2D reshaping needed)
        self.is_convolutional = False
        # initialize weights and biases
        self.initialize(None, None)
    
    def initialize(self, weights_initializer, bias_initializer):
        # disregard initializers and always set weights to 1 and bias to 0
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)
    
    def reformat(self, tensor):
        if tensor.ndim == 4:
            # Convert from (batch_size, channels, height, width) to (batch_size * height * width, channels)
            # transpose switches channels to last dimension
            self.tensor_shape = tensor.shape
            tensor = tensor.transpose(0, 2, 3, 1).reshape(-1, self.tensor_shape[1])
        elif tensor.ndim == 2:
            # Convert from (batch_size * height * width, channels) to (batch_size, channels, height, width)
            # transpose needed to switch channels back to second dimension
            batch_size, channels, height, width = self.tensor_shape
            tensor = tensor.reshape(batch_size, height, width, channels).transpose(0, 3, 1, 2)
        # return the (reshaped) tensor
        return tensor
    
    def forward(self, input_tensor):
        self.is_convolutional = (input_tensor.ndim == 4)
        # reshape tensor to 2D
        if self.is_convolutional:
            input_tensor = self.reformat(input_tensor)
        # store input tensor for backward pass
        self.input_tensor = input_tensor
        # decide phase for normalization
        if self.testing_phase:
            # in testing phase the training sets mean and variance are used
            # new computation only if no training has been done
            if self.moving_mean is None or self.moving_variance is None:
                # mean and variance for current batch
                self.mean = np.mean(input_tensor, axis=0)
                self.variance = np.var(input_tensor, axis=0)
                # set moving mean and variance accordingly
                self.moving_mean = self.mean
                self.moving_variance = self.variance
            else:
                # see formula in 3 1 1 slide 9
                self.mean = self.moving_mean
                self.variance = self.moving_variance
        else:
            # mean and variance for current batch
            self.mean = np.mean(input_tensor, axis=0)
            self.variance = np.var(input_tensor, axis=0)
            if self.moving_mean is None or self.moving_variance is None:
                # set moving mean and variance
                self.moving_mean = self.mean
                self.moving_variance = self.variance
            else:
                # see formula in 3 1 1 slide 9 for update
                self.moving_mean = self.momentum * self.moving_mean + (1 - self.momentum) * self.mean
                self.moving_variance = self.momentum * self.moving_variance + (1 - self.momentum) * self.variance
        # store normalized input for backward pass
        self.normalized_input = (input_tensor - self.mean) / np.sqrt(self.variance + self.epsilon)
        output_tensor = self.weights * self.normalized_input + self.bias
        # reshape tensor to 4D
        if self.is_convolutional:
            output_tensor = self.reformat(output_tensor)
        return output_tensor

    def backward(self, error_tensor):
        if self.is_convolutional:
            error_tensor = self.reformat(error_tensor)
        # compute the gradients w.r.t. weights and bias
        # formula: dL/dw = sum(E * norm_x)
        #          dL/db = sum(E)
        self.gradient_weights = np.sum(error_tensor * self.normalized_input, axis=0)
        self.gradient_bias = np.sum(error_tensor, axis=0)
        # compute input gradient dL/dx = E * w
        gradient_input = error_tensor * self.weights
        # use helper function to compute gradients w.r.t. input tensor
        gradient_tensor = compute_bn_gradients(gradient_input, self.input_tensor, self.weights, self.mean, self.variance, self.epsilon)
        # Optimize weights and biases if optimizers are defined
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias = self.optimizer.calculate_update(self.bias, self.gradient_bias)
        # format the the output tensor to correct shape
        if self.is_convolutional:
            gradient_tensor = self.reformat(gradient_tensor)
        return gradient_tensor