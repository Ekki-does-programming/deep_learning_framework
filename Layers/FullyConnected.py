import numpy as np

from .Base import BaseLayer

# Initialization of Fully Connected Layer
class FullyConnected(BaseLayer):
    # this class inherits the base layer

    def __init__(self, input_size, output_size):
        # calling super constructor
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        # initialize weights and bias as one matrix -> input_size + 1
        self.weights = np.random.uniform(low=0.0, high=1.0, size=(self.input_size + 1, self.output_size))

        self._optimizer = None
        # need gradients for w
        self._gradient_weights = None

    # methods
    def forward(self, input_tensor):
        """
        returns: input_tensor
        input: input tensor: matrix with shape (rows=batch_size, columns=input_size)
        """
        # see formula in slide set: y = x * w, where x also includes ones for the bias
        # add bias to input tensor
        input_tensor = np.hstack((input_tensor, np.ones((input_tensor.shape[0], 1)))) # add 1 to each x-vector
        self.input_tensor = input_tensor # needed for backward pass
        return input_tensor @ self.weights

    def backward(self, error_tensor):
        """
        returns: error_tensor
        input: error tensor: matrix with shape (rows=batch_size, columns=output_size)
        """
        # see formula in slide set: error = error * w^T
        # calculate gradient
        self._gradient_weights = self.input_tensor.T @ error_tensor
        # update weights only if trainable and if optimizer is set
        if self.trainable and self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self._gradient_weights)
        # exclude bias from error tensor
        return (error_tensor @ self.weights.T)[:,:-1]

    def initialize(self, weights_initializer, bias_initializer):
        """
        input: weights initializer
               bias initializer
        """
        self.weights = weights_initializer.initialize((self.input_size, self.output_size), self.input_size, self.output_size)
        self.weights = np.vstack((self.weights, bias_initializer.initialize((1, self.output_size), self.input_size, self.output_size)))

    # optimizer is a property of the class
    def _get_optimizer(self):
        return self._optimizer

    def _set_optimizer(self, opt):
        self._optimizer = opt

    def _del_optimizer(self):
        pass

    optimizer = property(
        fget = _get_optimizer,
        fset = _set_optimizer,
        fdel = _del_optimizer,
        doc = "The optimizer for the layer"
    )

    # property: gradient_weights
    @property
    def gradient_weights(self):
        return self._gradient_weights
