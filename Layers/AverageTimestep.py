import numpy as np


class AverageTimestep:
    """
    Averages the RNN output across all timesteps, instead of using only
    the last one. Lets every word's contribution reach the loss directly,
    rather than requiring it to survive the full recurrent BPTT chain.

    Forward
    -------
    Input  : (T, output_size)   — all RNN outputs across time
    Output : (1, output_size)   — mean over T

    Backward
    --------
    Input  : (1, output_size)   — error at the pooled output
    Output : (T, output_size)   — error/T, spread evenly across all timesteps
    """

    def __init__(self):
        self.trainable = False
        self.T = None  # sequence length, saved during forward for backward

    def forward(self, input_tensor):
        self.T = input_tensor.shape[0]
        return input_tensor.mean(axis=0, keepdims=True)

    def backward(self, error_tensor):
        return np.repeat(error_tensor / self.T, self.T, axis=0)
