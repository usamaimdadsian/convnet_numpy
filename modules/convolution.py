
import numpy as np


class Conv2D:
    '''
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        """
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output(aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel(both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        self.bias = np.zeros(self.out_channels)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        """
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters. Save necessary variables in self.cache for backward pass
        """
        out = None
        #############################################################################
        # TODO: Implement the convolution forward pass.                             #
        # Hint: 1) You may use np.pad for padding.                                  #
        #       2) You may implement the convolution with loops                     #
        #############################################################################
       
        N, C, H, W = x.shape
        F, C, HH, WW = self.weight.shape
        stride, pad = self.stride, self.padding

        H_prime = 1 + (H + 2 * pad - HH) // stride
        W_prime = 1 + (W + 2 * pad - WW) // stride
        out = np.zeros((N, F, H_prime, W_prime))
        

        x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
        for i in range(H_prime):
            for j in range(W_prime):
                x_slice = x_pad[:, :, i * stride:i * stride + HH, j * stride:j * stride + WW]
                for f in range(F):
                    out[:, f, i, j] = np.sum(x_slice * self.weight[f], axis=(1, 2, 3)) + self.bias[f]


        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = x
        return out

    def backward(self, dout):
        """
        The backward pass of convolution
        :param dout: upstream gradients
        :return: nothing but dx, dw, and db of self should be updated
        """
        x = self.cache
        #############################################################################
        # TODO: Implement the convolution backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the convolution with loops                     #
        #       2) don't forget padding when computing dx                           #
        #############################################################################
        N, C, H, W = x.shape
        F, C, HH, WW = self.weight.shape
        stride, pad = self.stride, self.padding

        H_prime = 1 + (H + 2 * pad - HH) // stride
        W_prime = 1 + (W + 2 * pad - WW) // stride
        dx = np.zeros_like(x)
        dx_pad = np.zeros((N, C, H + 2 * pad, W + 2 * pad))
        dw = np.zeros_like(self.weight)
        db = np.zeros_like(self.bias)

        x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
        for i in range(H_prime):
            for j in range(W_prime):
                x_slice = x_pad[:, :, i * stride:i * stride + HH, j * stride:j * stride + WW]
                for f in range(F):
                    dw[f] += np.sum(dout[:, f, i, j][:, np.newaxis, np.newaxis, np.newaxis] * x_slice, axis=0)
                    db[f] += np.sum(dout[:, f, i, j])
                    dx_pad[:, :, i * stride:i * stride + HH, j * stride:j * stride + WW] += dout[:, f, i, j][:, np.newaxis, np.newaxis, np.newaxis] * self.weight[f]

        dx = dx_pad[:, :, pad:-pad, pad:-pad]

        self.dx = dx
        self.dw = dw
        self.db = db
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
