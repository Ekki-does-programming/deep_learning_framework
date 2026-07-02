import numpy as np
from Layers import RNN, SoftMax, LSTM_improved
from Optimization.Optimizers import Adam
from Optimization.Constraints import L2_Regularizer
from Layers.Initializers import He
from NeuralNetwork import NeuralNetwork
from Optimization.Loss import CrossEntropyLoss
from Layers.AverageTimestep import AverageTimestep


class _Identity:
    """No-op passthrough, used to bypass a cell's built-in output Sigmoid
    activation — stacking it under SoftMax would cap output confidence
    and saturate gradients."""

    def forward(self, input_tensor):
        return input_tensor

    def backward(self, error_tensor):
        return error_tensor


# index of the built-in output-activation layer to bypass, per cell type —
# RNN's layers end in [FC, TanH, FC, Sigmoid]; LSTM_improved's end in
# [..gates.., FC, Sigmoid] with the gates taking up indices 0-5
_OUTPUT_ACTIVATION_INDEX = {"rnn": 3, "lstm": 7}


def build(
    cell_type: str = "rnn",  # "rnn" or "lstm"
    input_size: int = 50,    # must match GloVe embedding dim
    hidden_size: int = 64,   # recurrent hidden state size — increase for more capacity
    output_size: int = 2,    # positive / negative
):
    """
    Simple sentiment classifier using a recurrent cell:

        RNN or LSTM(input_size, hidden_size, output_size=2)
        -> AverageTimestep      [averages all timesteps: (T,2) -> (1,2)]
        -> SoftMax              [(1,2) class probabilities]

    Note: the recurrent cell already contains FC(hidden->output), so no need to
    add another FC layer here — its built-in Sigmoid is bypassed (see _Identity
    above) since stacking it under SoftMax would bound output confidence and
    saturate gradients.

    One review is processed per iteration (batch_size=1 in the data layer),
    with the sequence length T = max_seq_len.
    """
    if cell_type not in _OUTPUT_ACTIVATION_INDEX:
        raise ValueError(f"cell_type must be one of {list(_OUTPUT_ACTIVATION_INDEX)}, got {cell_type!r}")

    optimizer = Adam(learning_rate=1e-3, mu=0.9, rho=0.999)
    optimizer.add_regularizer(L2_Regularizer(1e-4))

    model = NeuralNetwork(optimizer, He(), He())

    # recurrent cell processes the full sequence (T, embedding_dim) -> (T, output_size)
    # memorize=False resets the hidden state between reviews
    if cell_type == "rnn":
        cell = RNN.RNN(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    else:
        cell = LSTM_improved.LSTM_improved(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    cell.memorize = False
    cell.layers[_OUTPUT_ACTIVATION_INDEX[cell_type]] = _Identity()  # bypass built-in output Sigmoid
    model.append_layer(cell)

    # Average over all timesteps: (T, 2) -> (1, 2)
    model.append_layer(AverageTimestep())

    # Normalise to probabilities for CrossEntropyLoss
    model.append_layer(SoftMax.SoftMax())

    model.loss_layer = CrossEntropyLoss()
    return model
