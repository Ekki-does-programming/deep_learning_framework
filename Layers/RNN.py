import numpy as np
import copy

from .FullyConnected import FullyConnected
from .TanH import TanH
from .Sigmoid import Sigmoid

def _clip_by_norm(gradient, max_norm):
    # rescale (not truncate) so direction is preserved — standard gradient-norm clipping
    norm = np.linalg.norm(gradient)
    if norm > max_norm:
        gradient = gradient * (max_norm / (norm + 1e-8))
    return gradient

# Elman Recurrent Neural Network
class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        # layer functionality
        self.trainable = True
        self.testing_phase = False
        # configuration
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_grad_norm = 5.0  # clip accumulated BPTT gradients to this L2 norm
        # optimizer
        self._optimizer = None
        self._output_optimizer = None
        # elman cell setup
        self.layers = [
            FullyConnected(input_size + hidden_size, hidden_size),
            TanH(),
            FullyConnected(hidden_size, output_size),
            Sigmoid()
        ]
        # memory
        self._memorize = False
        # hidden state
        self._gradient_weights = None
        self.x = None
        self.u = None
        self.h = None
        self.o = None
        self.y = None
        
    @property
    def weights(self):
        return self.layers[0].weights
    @weights.setter
    def weights(self, weights):
        self.layers[0].weights = weights

    @property
    def gradient_weights(self):
        return self._gradient_weights
    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights
    
    @property
    def memorize(self):
        return self._memorize
    @memorize.setter
    def memorize(self, value):
        self._memorize = value

    @property
    def optimizer(self):
        return self._optimizer
    @optimizer.setter
    def optimizer(self, optimizer):
        # update optimizer in elman cell
        self._optimizer = copy.deepcopy(optimizer)
        # separate optimizer instance for the hidden->output FC layer,
        # since it is not covered by the weights/gradient_weights properties above
        self._output_optimizer = copy.deepcopy(optimizer)
    
    def calculate_regularization_loss(self):
        if self.optimizer is not None and self.optimizer.regularizer is not None:
            return self.optimizer.regularizer.norm(self.weights)
    
    def initialize(self, weights_initializer, bias_initializer):
        # initialize all layer in elman cell
        self.layers[0].initialize(weights_initializer, bias_initializer)
        self.layers[2].initialize(weights_initializer, bias_initializer)
    
    def forward(self, input_tensor):
        # save input tensor
        input_tensor = input_tensor[:, None, :]
        # initialize hidden state
        if self.h is None:
            self.h = np.zeros((input_tensor.shape[0], 1, self.hidden_size))
        if self.memorize is False:
            self.h = np.zeros((input_tensor.shape[0], 1, self.hidden_size))
        self.x = np.zeros((input_tensor.shape[0], 1, self.input_size + self.hidden_size +1))
        self.u = np.zeros((input_tensor.shape[0], 1, self.hidden_size))
        self.o = np.zeros((input_tensor.shape[0], 1, self.hidden_size +1))
        self.y = np.zeros((input_tensor.shape[0], 1, self.output_size))
        # forward pass
        for t in range(input_tensor.shape[0]):
            # forward pass ref 3 2 slide 7
            self.u[t] = self.layers[0].forward(np.hstack((input_tensor[t], self.h[t-1])))
            self.h[t] = self.layers[1].forward(self.u[t])
            self.y[t] = self.layers[3].forward(self.layers[2].forward(self.h[t]))
            # save input tensor
            self.o[t] = self.layers[2].input_tensor
            self.x[t] = self.layers[0].input_tensor
        return self.y.squeeze(1)
    
    def backward(self, error_tensor):
        # initialize output tensor
        error_tensor = error_tensor[:, None, :]
        self.gradient_weights = np.zeros(self.weights.shape)
        output_gradient_weights = np.zeros(self.layers[2].weights.shape)
        error = np.zeros((error_tensor.shape[0], 1, self.input_size))
        hidden_error = np.zeros((1, self.hidden_size))
        # traverse the layers in reverse
        for t in reversed(range(error_tensor.shape[0])):
            # restore state
            self.layers[3].activation = self.y[t]
            self.layers[2].input_tensor = self.o[t]
            self.layers[1].activation = self.h[t]
            self.layers[0].input_tensor = self.x[t]
            # perform backward pass
            l3_error = self.layers[3].backward(error_tensor[t])
            l2_error = self.layers[2].backward(l3_error)
            l1_error = self.layers[1].backward(l2_error + hidden_error)
            l0_error = self.layers[0].backward(l1_error)
            # sum gradients
            self.gradient_weights += self.layers[0].gradient_weights
            output_gradient_weights += self.layers[2].gradient_weights
            # save hidden state
            error[t] = l0_error[:, :self.input_size]
            hidden_error = l0_error[:, self.input_size:]
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(
                self.weights, _clip_by_norm(self.gradient_weights, self.max_grad_norm)
            )
        if self._output_optimizer is not None:
            self.layers[2].weights = self._output_optimizer.calculate_update(
                self.layers[2].weights, _clip_by_norm(output_gradient_weights, self.max_grad_norm)
            )
        return error.squeeze(1)