import numpy as np
import matplotlib.pyplot as plt

import Layers.Base as Base

# Rectified Linear Unit
class ReLU(Base.BaseLayer):
    # this class inherits the base layer

    def __init__(self):
        super().__init__()
        self.prediction_tensor = None

    def forward(self, input_tensor):
        """
        returns: input_tensor
        input: input tensor: matrix with shape (rows=batch_size, columns=input_size)
        """
        # calculate y = max(0, x)
        self.prediction_tensor = np.maximum(0, input_tensor)
        return self.prediction_tensor

    def backward(self, error_tensor):
        """
        returns: error_tensor
        input: error tensor: matrix with shape (rows=batch_size, columns=output_size)
        """
        # create  the mask of active ReLU units
        mask = self.prediction_tensor > 0
        # multiply by error tensor to get the correct gradient
        return_tensor = error_tensor * mask
        return return_tensor