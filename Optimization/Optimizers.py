import numpy as np

class Optimizer:
    def __init__(self) -> None:
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer

# 1. Basic Optimizer
class Sgd(Optimizer):

    # constructor
    def __init__(self, learning_rate: float):
        super().__init__()
        self.learning_rate = learning_rate

    # method: calc. update
    def calculate_update(self, weight_tensor, gradient_tensor):
        # apply the regularizer
        if self.regularizer is not None:
            weight_tensor -= self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        # see formula in slide set: w(k+1) = w(k) - eta * gradient
        return weight_tensor - self.learning_rate * gradient_tensor
    
# 2. Advanced Optimizers
class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        # optimization related intermediary
        self.v_tensor = None
    
    def calculate_update(self, weight_tensor, gradient_tensor):
        # apply the regularizer
        if self.regularizer is not None:
            weight_tensor -= self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        # initialize on first update
        if self.v_tensor is None:
            self.v_tensor = np.zeros_like(weight_tensor)
        # w(k+1) = momentum * w(k) - eta * gradient
        self.v_tensor = self.momentum_rate * self.v_tensor + gradient_tensor
        return weight_tensor - self.learning_rate * self.v_tensor

class Adam(Optimizer):
    def __init__(self, learning_rate, mu, rho):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu    # beta_1
        self.rho = rho  # beta_2
        # optimization related intermediaries
        self.v_tensor = None
        self.r_tensor = None
        self.k = 0
    
    def calculate_update(self, weight_tensor, gradient_tensor):
        # apply the regularizer
        if self.regularizer is not None:
            weight_tensor -= self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        # fetch machine eps for given dtype
        dtype = weight_tensor.dtype if isinstance(gradient_tensor, np.ndarray) else type(weight_tensor)
        eps = np.finfo(dtype).eps
        # initialize parameters on first timestep
        if self.k == 0:
            self.v_tensor = np.zeros_like(weight_tensor)
            self.r_tensor = np.zeros_like(weight_tensor)
        # parameter updates
        self.k += 1
        self.v_tensor = self.mu * self.v_tensor + (1-self.mu) * gradient_tensor
        self.r_tensor = self.rho * self.r_tensor + (1-self.rho) * np.power(gradient_tensor, 2)
        # bias correction
        vhat_tensor = self.v_tensor / (1 - np.power(self.mu, self.k))
        rhat_tensor = self.r_tensor / (1 - np.power(self.rho, self.k))
        # w(k+1) = w(k) - eta * v(k) / (sqrt(r(k)) + eps)
        return weight_tensor - self.learning_rate * (vhat_tensor / (np.sqrt(rhat_tensor) + eps))