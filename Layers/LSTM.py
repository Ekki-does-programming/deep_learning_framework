import numpy as np
import copy

from .FullyConnected import FullyConnected
from .TanH import TanH
from .Sigmoid import Sigmoid

# Long Short-Term Memory
class LSTM:
    def __init__(self, input_size, hidden_size, output_size):
        # layer functionality
        self.trainable = True
        self.testing_phase = False
        # configuration
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # optimizer
        self._optimizer = None
        self._output_optimizer = None
        # elman cell setup
        self.layers = [
            FullyConnected(self.input_size + 2*self.hidden_size, 3*self.hidden_size),
            TanH(),
            FullyConnected(2*self.hidden_size, self.output_size),
            Sigmoid()
        ]
        # print(self.input_size + 2*self.hidden_size, 3*self.hidden_size)
        # memory
        self._memorize = False
        # hidden state
        self._gradient_weights = None
        self.x = None
        self.u = None
        self.h = None
        self.o = None
        self.y = None
        self.c = None
        
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
            self.h = np.zeros((input_tensor.shape[0], 1, 2*self.hidden_size))
            self.c = np.zeros((input_tensor.shape[0], 1, self.hidden_size))
        if self.memorize is False:
            self.h = np.zeros((input_tensor.shape[0], 1, 2*self.hidden_size))
            self.c = np.zeros((input_tensor.shape[0], 1, self.hidden_size))
        self.x = np.zeros((input_tensor.shape[0], 1, self.input_size + 2*self.hidden_size +1))
        self.u = np.zeros((input_tensor.shape[0], 1, 3*self.hidden_size))
        self.o = np.zeros((input_tensor.shape[0], 1, 2*self.hidden_size +1))
        self.y = np.zeros((input_tensor.shape[0], 1, self.output_size))
        # forward pass
        for t in range(input_tensor.shape[0]):
            # forward pass ref 3 2 slide 7
            self.u[t] = self.layers[0].forward(np.hstack((input_tensor[t], self.h[t-1])))
            temp = self.layers[1].forward(self.u[t])
            self.h[t] = temp[:, :2*self.hidden_size]
            self.c[t] = temp[:, 2*self.hidden_size:]
            self.y[t] = self.layers[3].forward(self.layers[2].forward(self.h[t]))
            # save input tensor
            self.o[t] = self.layers[2].input_tensor
            self.x[t] = self.layers[0].input_tensor
        return self.y.squeeze(1)
    
    def backward(self, error_tensor):
        # initialize output tensor
        error_tensor = error_tensor[:, None, :]
        self.gradient_weights = np.zeros(self.weights.shape)
        error = np.zeros((error_tensor.shape[0], 1, self.input_size))
        hidden_error = np.zeros((1, 2*self.hidden_size))
        c_error = np.zeros((1, self.hidden_size))
        # traverse the layers in reverse
        for t in reversed(range(error_tensor.shape[0])):
            # restore state
            self.layers[3].activation = self.y[t]
            self.layers[2].input_tensor = self.o[t]
            self.layers[1].activation = np.hstack((self.h[t], self.c[t]))
            self.layers[0].input_tensor = self.x[t]
            # perform backward pass
            l3_error = self.layers[3].backward(error_tensor[t])
            l2_error = self.layers[2].backward(l3_error)
            l1_error = self.layers[1].backward(np.hstack((l2_error + hidden_error, c_error)))
            l0_error = self.layers[0].backward(l1_error)
            # sum gradients
            self.gradient_weights += self.layers[0].gradient_weights
            # save hidden state
            error[t] = l0_error[:, :self.input_size]
            hidden_error = l0_error[:, self.input_size:]
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
        if self._output_optimizer is not None:
            self.layers[2].weights = self._output_optimizer.calculate_update(
                self.layers[2].weights, self.layers[2].gradient_weights
            )
        return error.squeeze(1)