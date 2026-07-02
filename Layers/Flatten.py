import numpy as np

import Layers.Base as Base

# Flatten Layer
class Flatten(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        # save for backward
        self.input_shape = None

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape # = (batch_size, height, width, channels)
        # reshape to (batch_size, height * width * channels); "-1" is placeholder for the other dimension
        return input_tensor.reshape(self.input_shape[0], -1)

    def backward(self, error_tensor):
        return error_tensor.reshape(self.input_shape)