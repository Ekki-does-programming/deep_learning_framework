import numpy as np
from scipy.signal import correlate, convolve
import copy

from .Base import BaseLayer

# Convolutional Layer
class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True
        self.stride_shape = np.tile(stride_shape, 2)[:2] # ALWAYS TUPLE:  (X, Y) OR (X, X)
        self.convolution_shape = convolution_shape # 1D: [c, m]; 2D: [c, m, n]; channels, columns_kernel, rows_kernel
        self.num_kernels = num_kernels # int
        # build the correct slice mechanism for the dimensionalities
        if len(convolution_shape) == 2: # 1D case
            # view generation
            self.slicer = [slice(None, None, self.stride_shape[0])]
            self.bias_slicer = [np.newaxis]
            # axis remover
            self.squeeze_axis = (3, 4)
            # tensordot axis
            self.forward_dot_axis = ((1, -1), (1, 2))
            self.backward_gradient_dot_axis = ((0, -1), (0, -1))
            # reshape shapes
            self.forward_shaper = (0, 2, 1)
            self.backward_gradient_shaper = (-1, 0, 1)
            # kernel flipping
            self.kernel_flip = tuple([-1])
        else: # 2D case
            # view generation
            self.slicer = [slice(None, None, self.stride_shape[0]), slice(None, None, self.stride_shape[1])]
            self.bias_slicer = [np.newaxis, np.newaxis]
            # axis remover
            self.squeeze_axis = (4, 5)
            # tensordot axis
            self.forward_dot_axis = ((1, -2, -1), (1, 2, 3))
            self.backward_gradient_dot_axis = ((0, -2, -1), (0, -2, -1))
            # reshape shapes
            self.forward_shaper = (0, 3, 1, 2)
            self.backward_gradient_shaper = (-1, 0, 1, 2)
            # kernel flipping
            self.kernel_flip = (-2, -1)
        # output shape
        self.output_shape = None
        # weights and biases uniformly initialized
        self.weights = np.random.uniform(low=0.0, high=1.0, size=(num_kernels, *convolution_shape))
        self.bias = np.random.uniform(low=0.0, high=1.0, size=(num_kernels,))
        # gradients
        self.gradient_weights = None
        self.gradient_bias = None
        # save for backward
        self.prediction_tensor = None
        # optimizers for weights and biases
        self._optimizer_weights = None
        self._optimizer_bias = None
        self._optimizer = None

    # property optimizer: store optimizers for weights and biases
    def _get_optimizer_weights(self):
        return self._optimizer_weights
    def _set_optimizer_weights(self, opt):
        self._optimizer_weights = copy.deepcopy(opt)
    def _del_optimizer_weights(self):
        pass

    optimizer_weights = property(
        fget = _get_optimizer_weights,
        fset = _set_optimizer_weights,
        fdel = _del_optimizer_weights,
        doc = "The optimizer for the layer for the weights"
    )

    def _get_optimizer_bias(self):
        return self._optimizer_bias
    def _set_optimizer_bias(self, opt):
        self._optimizer_bias = copy.deepcopy(opt)
    def _del_optimizer_bias(self):
        pass

    optimizer_bias = property(
        fget = _get_optimizer_bias,
        fset = _set_optimizer_bias,
        fdel = _del_optimizer_bias,
        doc = "The optimizer for the layer for the bias"
    )

    def _get_optimizer(self):
        return self._optimizer
    def _set_optimizer(self, opt):
        self._optimizer = opt
        self._optimizer_weights = copy.deepcopy(opt)
        self._optimizer_bias = copy.deepcopy(opt)
    def _del_optimizer(self):
        pass

    optimizer = property(
        fget = _get_optimizer,
        fset = _set_optimizer,
        fdel = _del_optimizer,
        doc = "The optimizer for the layer"
    )

    # helper functions
    def _pad_tensor(self, input_tensor, kernel_shape):
        """
        Apply zero-padding to a tensor
        """
        padded_tensor = None
        # uneven: P = int((K-1)/2); even: P = K // 2, P = K // 2 - 1
        # 1. case: asymmetric padding along single axis,
        # 2. case symmetric padding along single axis -> (front_pad, end_pad)
        padding = [(k // 2, k // 2 - 1) if k % 2 == 0 else (int((k - 1) / 2), int((k - 1) / 2)) for k in kernel_shape]
        if len(input_tensor.shape) == 3:  # 1D case
            padded_tensor = np.pad(input_tensor, ((0, 0), (0, 0), padding[0]), mode='constant', constant_values=0)
        elif len(input_tensor.shape) == 4:  # 2D case
            padded_tensor = np.pad(input_tensor, ((0, 0), (0, 0), padding[0], padding[1]), mode='constant',constant_values=0)
        return padded_tensor

    def forward(self, input_tensor):
        """
        returns: input_tensor for next layer
        input layout: 1D: (batch_size, channel_size, y)
                      2D: (batch_size, channel_size, y, x)
        """
        self.input_tensor = input_tensor

        # Calculate output_shape
        (batch_size, channel_size, *input_shape) = input_tensor.shape
        output_shape = np.int32(np.ceil(np.array(input_shape) / self.stride_shape[:len(input_shape)]))
        self.output_shape = (batch_size, self.num_kernels, *output_shape)

        # Zero padding of input_tensor
        _, *kernel_shape = self.convolution_shape
        self.padded_input_tensor = self._pad_tensor(input_tensor, kernel_shape)

        # Initialize output tensor (only for the other possibilities)
        #output_tensor = np.zeros(self.output_shape)

        # ========== CONVOLUTION ==========
        # create a sliding window for convolution
        input_windows = np.lib.stride_tricks.sliding_window_view(self.padded_input_tensor, (1, 1, *kernel_shape))
        # remove empty dimensions
        input_windows = input_windows.squeeze(axis=self.squeeze_axis)
        # perform convolution
        output_tensor = np.tensordot(input_windows, self.weights, axes=self.forward_dot_axis)
        # reshape to (B, K, M, N)
        output_tensor = output_tensor.transpose(*self.forward_shaper)
        # wasteful subsampling [:, :, ::stride_shape...], automatic dimension switching
        output_tensor = output_tensor[tuple([slice(None), slice(None)] + self.slicer)]
        # add bias [None, :, None...], automatic dimension switching
        output_tensor += self.bias[tuple([np.newaxis, slice(None)] + self.bias_slicer)]
        #  ========== OTHER POSSIBILITY 1 ==========
        # convolution implementation
        # output_tensor = np.zeros((batch_size, self.num_kernels, *output_shape))
        # for i in range(padded_input_tensor.shape[0]):
        #     for k in range(self.weights.shape[0]):
        #         output_tensor[i, k, ...] = correlate(padded_input_tensor[i], self.weights[k], mode='valid')[tuple([slice(None)] + self.slicer)] + self.bias[k]
        # ========== OTHER POSSIBILITY 2 ==========
        #  loop over every element in BATCH
        # for i, image in enumerate(padded_input_tensor):
        #    # loop over every kernel
        #    for k, kernel in enumerate(self.weights):
        #        # calculate correlation
        #        out = correlate(image, kernel, mode='valid')
        #        # wasteful striding
        #        out = out[:, ::self.stride_shape[0]]    # :: = start:end:step
        #        if len(self.stride_shape) > 1:
        #            out = out[:, :, ::self.stride_shape[1]]
        #        # add bias (element-wise, see comment 3d)
        #        out = np.add(out, self.bias[k])
        #        # save to output tensor
        #        output_tensor[i, k, ...] = out
        #  ========== OTHER POSSIBILITY 3 ==========
        # Perform convolution
        # if len(input_shape) == 1:  # 1D case
        #     for b in range(batch_size):
        #         for k in range(self.num_kernels):
        #             for c in range(channel_size):
        #                 for i in range(0, input_shape[0], self.stride_shape[0]):
        #                     output_tensor[b, k, i // self.stride_shape[0]] += np.sum(
        #                         self.padded_input_tensor[b, c, i:i + kernel_shape[0]] * self.weights[k, c]
        #                     )
        #             output_tensor[b, k] += self.bias[k]
        # elif len(input_shape) == 2:  # 2D case
        #     for b in range(batch_size):
        #         for k in range(self.num_kernels):
        #             for c in range(channel_size):
        #                 for i in range(0, input_shape[0], self.stride_shape[0]):
        #                     for j in range(0, input_shape[1], self.stride_shape[1]):
        #                         output_tensor[b, k, i // self.stride_shape[0], j // self.stride_shape[1]] += np.sum(
        #                             self.padded_input_tensor[b, c, i:i + kernel_shape[0], j:j + kernel_shape[1]] * self.weights[k, c]
        #                         )
        #             output_tensor[b, k] += self.bias[k]
        return output_tensor

    def backward(self, error_tensor):
        """
        Updates the parameters using the optimizer (if available)
        and returns the error tensor.

        Info:
            1. Use convolution in backward pass
            2. Handle bias; sum over (B, W, H) E_n
            3. Use same formulas as for FC layer
                - Filters need to be flipped (180 degree rotation)
                - Rearrange weights (had H kernels and S channels -> now need S kernels)
            4. Gradient w. r. t. layers: Rearrange kernels (H x K_{S, N, M} -> S x K^hat_{H, N, M})
            5. Gradient w. r. t. X: X_s * E_{h,n} -> grad K_{h, S, N, M}
        """
        # Handle BIAS; sum over (B, W, H) E_n
        _ndims = np.arange(len(error_tensor.shape))
        self.gradient_bias = np.sum(error_tensor, axis=(_ndims[0], *_ndims[2:]))

        # Gradient with respect to lower LAYERS
        # build S new kernels of height H and width m x n
        _ndims = np.arange(len(self.weights.shape))
        new_kernels = np.transpose(self.weights, (_ndims[1], _ndims[0], *_ndims[2:]))
        # ========== OTHER POSSIBILITY ==========
        # new_kernels = np.zeros((self.convolution_shape[0], self.num_kernels, *self.convolution_shape[1:]))
        # for k, kernel in enumerate(self.weights):
        #     for c in range(self.convolution_shape[0]):
        #         new_kernels[c, k] = kernel[c]

        # upsampling of error tensor to match the input tensor (before striding)
        shape = np.array(self.input_tensor.shape)
        shape[1] = self.num_kernels     # B, K, W, H (b, k, m, n)
        upsampled_error_tensor = np.zeros(tuple(shape))
        # cases for 1D and 2D
        upsampled_error_tensor[tuple([slice(None), slice(None)] + self.slicer)] = error_tensor
        #if len(self.convolution_shape) == 2:
            #upsampled_error_tensor[:, :, ::self.stride_shape[0]] = error_tensor
        #else:
            # case if stride_shape is single value
            #stride_shape_dim1 = self.stride_shape[0]
            #if len(self.stride_shape) > 1:
            #    stride_shape_dim1 = self.stride_shape[1]
            #upsampled_error_tensor[:, :, ::self.stride_shape[0], ::stride_shape_dim1] = error_tensor
        
        # initialize output tensor (needed for other possibilities)
        # output_tensor = np.zeros_like(self.input_tensor)
        
        # convolve new kernels with upsampled error tensor
        padded_error_tensor = self._pad_tensor(upsampled_error_tensor, new_kernels.shape[2:])
        # flip kernels -> since tensordot is correlation instead of convolution
        flipped_kernels = np.flip(new_kernels, axis=self.kernel_flip)
        # create a sliding window for convolution
        error_windows = np.lib.stride_tricks.sliding_window_view(padded_error_tensor, (1, 1, *new_kernels.shape[2:]))
        # remove empty dimensions
        error_windows = error_windows.squeeze(axis=self.squeeze_axis)
        # perform the convolution
        temp = np.tensordot(error_windows, flipped_kernels, axes=self.forward_dot_axis)
        # reshape to (B, C, M...), automatic dimension switching
        output_tensor = temp.transpose(*self.forward_shaper)
        # ========== OTHER POSSIBILITY 1 ==========
        # convolve new kernels with upsampled error tensor
        # for i, i_err in enumerate(upsampled_error_tensor):
        #     for k, nk in enumerate(new_kernels):
        #         for c in range(self.num_kernels):
        #             output_tensor[i, k] += convolve(i_err[c], nk[c], mode='same')
        # ========== OTHER POSSIBILITY 2 ==========
        # convolve new kernels with upsampled error tensor
        # for b in range(padded_error_tensor.shape[0]):
        #     for k in range(flipped_kernels.shape[0]):
        #         for c in range(self.num_kernels):
        #             output_tensor[b, k] += correlate(padded_error_tensor[b, c], flipped_kernels[k, c], mode='valid')

        # initialize gradient weights (needed for other possibilities)
        # self.gradient_weights = np.zeros_like(self.weights) # shape: (H, C, M, N)

        # ========== GRADIENT w. r. t. LAYERS ==========
        # build the sliding window for computation
        input_windows = np.lib.stride_tricks.sliding_window_view(self.padded_input_tensor, (1, 1, *upsampled_error_tensor.shape[2:]))
        # remove empty dimensions
        input_windows = input_windows.squeeze(axis=self.squeeze_axis)
        # perform the convolution
        self.gradient_weights = np.tensordot(input_windows, upsampled_error_tensor, axes=self.backward_gradient_dot_axis)
        # reshape to (H, C, M...), automatic dimension switching
        self.gradient_weights = self.gradient_weights.transpose(self.backward_gradient_shaper)
        # ========== OTHER POSSIBILITY 1 ==========
        # for i, image in enumerate(self.padded_input_tensor):
        #     # select error tensor from same batch
        #     err_tensor = upsampled_error_tensor[i]
        #     for h, err_layer in enumerate(err_tensor):
        #         for c, channel in enumerate(image):
        #             # calculate correlation
        #             self.gradient_weights[h, c] += correlate(channel, err_layer, mode='valid')
        # ========== OTHER POSSIBILITY 2 ==========
        # Gradient w. r. t. X: X_s * E_{h,n} -> grad K_{h, S, N, M} --> KERNELS/ WEIGHTS
        # if len(self.gradient_weights[2:]) == 1:
        #    for b in range(self.padded_input_tensor.shape[0]):
        #        for h in range(upsampled_error_tensor.shape[1]):
        #            for c in range(self.padded_input_tensor.shape[1]):
        #                for m in range(self.gradient_weights.shape[2]):
        #                    err_tensor = upsampled_error_tensor[b, h]
        #                    pin_tensor = self.padded_input_tensor[b, c]
        #                    sum = np.sum(pin_tensor[m:m + err_tensor.shape[0]] * err_tensor)
        #                    self.gradient_weights[h, c, m] += sum
        # elif len(self.gradient_weights[2:]) == 2:
        #     for b in range(self.padded_input_tensor.shape[0]):
        #         for h in range(upsampled_error_tensor.shape[1]):
        #             for c in range(self.padded_input_tensor.shape[1]):
        #                 for m in range(self.gradient_weights.shape[2]):
        #                     for n in range(self.gradient_weights.shape[3]):
        #                         err_tensor = upsampled_error_tensor[b, h]
        #                         pin_tensor = self.padded_input_tensor[b, c]
        #                         sum = np.sum(pin_tensor[m:m + err_tensor.shape[0], n:n + err_tensor.shape[1]] * err_tensor)
        #                         self.gradient_weights[h, c, m, n] += sum

        # apply optimizer if set
        if self._optimizer_weights is not None:
            self.weights = self._optimizer_weights.calculate_update(self.weights, self.gradient_weights)
        if self._optimizer_bias is not None:
            self.bias = self._optimizer_bias.calculate_update(self.bias, self.gradient_bias)

        return output_tensor

    def initialize(self, weights_initializer, bias_initializer):
        # compute the input and output sizes
        fan_in  = np.prod(self.convolution_shape)
        fan_out = np.prod((self.num_kernels, *self.convolution_shape[1:]))
        # update weights and bias
        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias    = bias_initializer.initialize(self.bias.shape, fan_in, fan_out)