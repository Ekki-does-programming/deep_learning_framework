import numpy as np
import matplotlib.pyplot as plt

# Cross Entropy Loss
class CrossEntropyLoss:

    def __init__(self):
        self.prediction_tensor = None
        self.eps = None

    def forward(self, prediction_tensor, label_tensor):
        """
        returns: loss
        input: prediction tensor: matrix with shape (rows=batch_size, columns=output_size)
               label tensor: matrix with shape (rows=batch_size, columns=output_size)
        """
        # Get dtype info for machine epsilon
        dtype = prediction_tensor.dtype
        self.eps = np.finfo(dtype).eps
        self.prediction_tensor = prediction_tensor + self.eps
        # compute the loss -1/N *sum(label * log(pred))
        loss = np.sum(-(label_tensor * np.log(self.prediction_tensor)))
        return loss

    def backward(self, label_tensor):
        """
        returns: error tensor: matrix with shape (rows=batch_size, columns=output_size)
        input: label tensor: matrix with shape (rows=batch_size, columns=output_size)
        """
        # En = -y / (yp + eps)
        error_tensor = - (label_tensor / self.prediction_tensor)
        return error_tensor