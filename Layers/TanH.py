import numpy as np

from .Base import BaseLayer

# Recurrent Layer Activation TanH
class TanH(BaseLayer):
    def __init__(self):
        super().__init__()
        # store the dynamic activations
        self.activation = None
    
    def forward(self, input_tensor):
        self.activation = np.tanh(input_tensor)
        return self.activation

    def backward(self, error_tensor):
        return error_tensor * (1 - self.activation ** 2)