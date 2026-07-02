import numpy as np
import numpy.lib.stride_tricks as st

import Layers.Base as Base

# Pooling Layer
class Pooling(Base.BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        # storing for backward
        self.prediction_tensor = None
        self.input_tensor = None
        self.indices_tensor = None

    def forward(self, input_tensor):
        """
        returns: input tensor: matrix of shape
        input: input tensor: matrix of shape (batch_size, height, width, channels)
        """
        (batch_size, channel_count, *input_shape) = input_tensor.shape
        input_shape = tuple(input_shape)
        output_shape = tuple( (a-b) // c +1 for (a,b,c) in zip(input_shape, self.pooling_shape, self.stride_shape))
        # initialize new input tensor
        self.prediction_tensor = np.zeros((batch_size, channel_count, output_shape[0], output_shape[1]))
        self.indices_tensor = np.zeros((batch_size, channel_count, output_shape[0], output_shape[1], 2), dtype=int)
        self.input_tensor = input_tensor
        # produce window across the pooling per index in the array
        window_tensor = np.lib.stride_tricks.sliding_window_view(self.input_tensor, (1, 1) + self.pooling_shape)[:, :, ::self.stride_shape[0], ::self.stride_shape[1]]
        # collapse the last dimensions
        window_tensor = window_tensor.reshape(*window_tensor.shape[:4], -1)
        # compute the local maxima position
        indices_ = np.argmax(window_tensor, axis=-1)
        row_indices = indices_ // self.pooling_shape[1]
        col_indices = indices_ %  self.pooling_shape[1]
        # compute global maxima with correct shaping and store back
        self.indices_tensor[..., 0] = row_indices + np.arange(output_shape[0])[:, None] * self.stride_shape[0]
        self.indices_tensor[..., 1] = col_indices + np.arange(output_shape[1]) * self.stride_shape[1]
        # compute prediction tensor
        self.prediction_tensor = window_tensor.max(axis=-1)
        return self.prediction_tensor
        

    def backward(self, error_tensor):
        """
        returns: error tensor: matrix of shape
        input: error tensor: matrix of shape
        """
        # Initialize the gradient for the input with zeros
        grad_input = np.zeros_like(self.input_tensor)
        # build the batch and channel index tensors
        b_row = np.arange(error_tensor.shape[0])[:, None, None, None]
        c_col = np.arange(error_tensor.shape[1])[None, :, None, None]
        # build the maximum row and column index tensors
        m_row = self.indices_tensor[..., 0]
        m_col = self.indices_tensor[..., 1]
        # backpropagate the error, avoid discard of repetitive entries of += by np.add.at
        np.add.at(grad_input, (b_row, c_col, m_row, m_col), error_tensor)
        return grad_input