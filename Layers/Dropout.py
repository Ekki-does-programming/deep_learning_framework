import numpy as np

from .Base import BaseLayer
# Inverted Dropout Regularization
class Dropout(BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability
        self.mask = None

    def forward(self, input_tensor):
        if self.testing_phase:
            # do not interact during testing
            return input_tensor
        else:
            # create new mask with entries either 0 or 1/p
            self.mask = 1 / self.probability * (np.random.rand(*input_tensor.shape) > 1 - self.probability)
            # apply the mask to the input
            return input_tensor * self.mask

    def backward(self, error_tensor):
        # return error tensor multiplied by the mask
        return error_tensor * self.mask