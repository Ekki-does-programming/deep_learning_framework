import numpy as np


class LastTimestep:
    """
    Extracts the last timestep from an RNN output sequence.

    Forward
    -------
    Input  : (T, output_size)   — all RNN outputs across time
    Output : (1, output_size)   — only the final timestep

    Backward
    --------
    Input  : (1, output_size)   — error at the last timestep
    Output : (T, output_size)   — error placed at t=T-1, zeros elsewhere

    This lets the network only classify based on what the RNN has
    seen after processing the full sequence.
    """

    def __init__(self):
        self.trainable = False
        self.T = None  # sequence length, saved during forward for backward

    def forward(self, input_tensor):
        self.T = input_tensor.shape[0]
        # return only the last row, keep 2-D shape (1, output_size)
        return input_tensor[[-1], :]

    def backward(self, error_tensor):
        # route the gradient only to the last timestep
        out = np.zeros((self.T, error_tensor.shape[1]))
        out[-1] = error_tensor[0]
        return out