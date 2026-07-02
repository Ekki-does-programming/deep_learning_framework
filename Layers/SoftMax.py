import numpy as np
import matplotlib.pyplot as plt

import Layers.Base as Base

# Soft Maximum
class SoftMax(Base.BaseLayer):
    # this class inherits the base layer

    def __init__(self):
        super().__init__()
        self.prediction_tensor = None

    def forward(self, input_tensor):
        """
        returns: input_tensor
        input: input tensor: matrix with shape (rows=batch_size, columns=input_size)
        """
        # axis=1 to split the tensor into the batches for the function calls
        # shift to avoid numerical instability of exp() series
        shifted_tensor = input_tensor - np.max(input_tensor, axis=1, keepdims=True)
        exponent_tensor = np.exp(shifted_tensor)
        # compute y = exp(x) / sum(exp(x))
        self.prediction_tensor = exponent_tensor / np.sum(exponent_tensor, axis=1, keepdims=True)
        return self.prediction_tensor
    
    def backward(self, error_tensor):
        """
        returns: error_tensor
        input: error tensor: matrix with shape (rows=batch_size, columns=output_size)
        """
        # multiply En*yn
        multiplied_tensor = np.multiply(error_tensor, self.prediction_tensor)
        # sum over all elements within a batch and subtract from En
        sum_tensor = error_tensor - np.sum(multiplied_tensor, axis=1, keepdims=True)
        # correct the gradient
        return_tensor = self.prediction_tensor * sum_tensor
        return return_tensor