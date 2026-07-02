import numpy as np

# Initializers Constant, UniformRandom, Xavier, He

"""
weights shape : tuple (output size, input size)
"""
"""
For FULLYCONNECTED Layers::
fan in  : input dimension of weights
fan out : output dimension of weights
"""
"""
For CONVOLUTIONAL Layers::
fan in  : [input channels  * kernel height * kernel width]
fan out : [output channels * kernel height * kernel width]
"""
class Constant:
    def __init__(self, constant = 0.1):
        self.constant = constant

    def initialize(self, weights_shape, fan_in, fan_out):
        """
        returns: tensor: matrix with shape
        """
        return_tensor = np.full(weights_shape, fill_value=self.constant)
        return return_tensor

class UniformRandom:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        """
        returns: tensor: matrix with shape
        """
        return_tensor = np.random.uniform(0, 1, weights_shape)
        return return_tensor

class Xavier:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        """
        returns: tensor: matrix with shape
        """
        sigma = np.sqrt(2 / (fan_out + fan_in))
        return_tensor = np.random.normal(0, sigma, weights_shape)
        return return_tensor

class He:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        """
        returns: tensor: matrix with shape
        """
        sigma = np.sqrt(2 / fan_in)
        return_tensor = np.random.normal(0, sigma, weights_shape)
        return return_tensor