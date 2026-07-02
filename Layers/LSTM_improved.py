import numpy as np
import copy

from .FullyConnected import FullyConnected
from .TanH import TanH
from .Sigmoid import Sigmoid
from .RNN import _clip_by_norm

# Long Short-Term Memory (gated recurrent cell)
class LSTM_improved:
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
        # lstm cell setup
        # layers[0] projects [x_t, h_(t-1)] -> 4 gate pre-activations, split in
        # order (forget, input, candidate, output); layers[1..4] activate them;
        # layers[5] squashes the cell state for the h_t = o_t * tanh(c_t) step;
        # layers[6..7] are the output projection, mirroring RNN.py
        self.layers = [
            FullyConnected(input_size + hidden_size, 4 * hidden_size),
            Sigmoid(),   # forget gate
            Sigmoid(),   # input gate
            TanH(),      # cell candidate
            Sigmoid(),   # output gate
            TanH(),      # tanh(c_t)
            FullyConnected(hidden_size, output_size),
            Sigmoid(),
        ]
        # memory
        self._memorize = False
        # hidden/cell state
        self._gradient_weights = None
        self.h = None
        self.c = None
        self.gate_input = None
        self.out_input = None
        self.f = None
        self.i = None
        self.g = None
        self.o_gate = None
        self.tanh_c = None
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
        # update optimizer in lstm cell
        self._optimizer = copy.deepcopy(optimizer)
        # separate optimizer instance for the hidden->output FC layer,
        # since it is not covered by the weights/gradient_weights properties above
        self._output_optimizer = copy.deepcopy(optimizer)

    def calculate_regularization_loss(self):
        if self.optimizer is not None and self.optimizer.regularizer is not None:
            return self.optimizer.regularizer.norm(self.weights)

    def initialize(self, weights_initializer, bias_initializer):
        # initialize all trainable layers in lstm cell
        self.layers[0].initialize(weights_initializer, bias_initializer)
        self.layers[6].initialize(weights_initializer, bias_initializer)

    def forward(self, input_tensor):
        input_tensor = input_tensor[:, None, :]
        T = input_tensor.shape[0]
        H = self.hidden_size

        # h/c use T+1 entries: index 0 is the state *before* t=0 (carried over
        # from the previous sequence if memorize=True, else zero)
        if self.memorize and self.h is not None:
            h_init = self.h[-1].copy()
            c_init = self.c[-1].copy()
        else:
            h_init = np.zeros((1, H))
            c_init = np.zeros((1, H))

        self.h = np.zeros((T + 1, 1, H))
        self.c = np.zeros((T + 1, 1, H))
        self.h[0] = h_init
        self.c[0] = c_init

        self.gate_input = np.zeros((T, 1, self.input_size + H + 1))
        self.out_input = np.zeros((T, 1, H + 1))
        self.f = np.zeros((T, 1, H))
        self.i = np.zeros((T, 1, H))
        self.g = np.zeros((T, 1, H))
        self.o_gate = np.zeros((T, 1, H))
        self.tanh_c = np.zeros((T, 1, H))
        self.y = np.zeros((T, 1, self.output_size))

        for t in range(T):
            h_prev, c_prev = self.h[t], self.c[t]

            z = np.hstack((input_tensor[t], h_prev))
            gate_pre = self.layers[0].forward(z)
            self.gate_input[t] = self.layers[0].input_tensor

            a_f, a_i, a_g, a_o = np.split(gate_pre, 4, axis=1)
            f_t = self.layers[1].forward(a_f)
            i_t = self.layers[2].forward(a_i)
            g_t = self.layers[3].forward(a_g)
            o_t = self.layers[4].forward(a_o)

            c_t = f_t * c_prev + i_t * g_t
            tanh_c_t = self.layers[5].forward(c_t)
            h_t = o_t * tanh_c_t

            y_pre = self.layers[6].forward(h_t)
            self.out_input[t] = self.layers[6].input_tensor
            y_t = self.layers[7].forward(y_pre)

            self.f[t], self.i[t], self.g[t], self.o_gate[t] = f_t, i_t, g_t, o_t
            self.tanh_c[t] = tanh_c_t
            self.h[t + 1] = h_t
            self.c[t + 1] = c_t
            self.y[t] = y_t

        return self.y.squeeze(1)

    def backward(self, error_tensor):
        error_tensor = error_tensor[:, None, :]
        T = error_tensor.shape[0]

        self.gradient_weights = np.zeros(self.weights.shape)
        output_gradient_weights = np.zeros(self.layers[6].weights.shape)
        error = np.zeros((T, 1, self.input_size))
        dh_next = np.zeros((1, self.hidden_size))
        dc_next = np.zeros((1, self.hidden_size))

        for t in reversed(range(T)):
            # ---- output path: y_t -> h_t ----
            self.layers[7].activation = self.y[t]
            d_y_pre = self.layers[7].backward(error_tensor[t])

            self.layers[6].input_tensor = self.out_input[t]
            dh = self.layers[6].backward(d_y_pre) + dh_next

            # ---- h_t = o_t * tanh(c_t) ----
            do_t = dh * self.tanh_c[t]
            d_tanh_c = dh * self.o_gate[t]

            self.layers[5].activation = self.tanh_c[t]
            dc = self.layers[5].backward(d_tanh_c) + dc_next

            # ---- c_t = f_t * c_(t-1) + i_t * g_t ----
            df_t = dc * self.c[t]      # self.c[t] holds c_(t-1) for this step
            dc_next = dc * self.f[t]   # becomes dc_(t-1) for the next iteration
            di_t = dc * self.g[t]
            dg_t = dc * self.i[t]

            self.layers[1].activation = self.f[t]
            da_f = self.layers[1].backward(df_t)

            self.layers[2].activation = self.i[t]
            da_i = self.layers[2].backward(di_t)

            self.layers[3].activation = self.g[t]
            da_g = self.layers[3].backward(dg_t)

            self.layers[4].activation = self.o_gate[t]
            da_o = self.layers[4].backward(do_t)

            d_gate_pre = np.hstack((da_f, da_i, da_g, da_o))

            self.layers[0].input_tensor = self.gate_input[t]
            dz = self.layers[0].backward(d_gate_pre)

            error[t] = dz[:, :self.input_size]
            dh_next = dz[:, self.input_size:]

            self.gradient_weights += self.layers[0].gradient_weights
            output_gradient_weights += self.layers[6].gradient_weights

        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(
                self.weights, _clip_by_norm(self.gradient_weights, self.max_grad_norm)
            )
        if self._output_optimizer is not None:
            self.layers[6].weights = self._output_optimizer.calculate_update(
                self.layers[6].weights, _clip_by_norm(output_gradient_weights, self.max_grad_norm)
            )
        return error.squeeze(1)