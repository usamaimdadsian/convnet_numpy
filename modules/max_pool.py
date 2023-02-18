
import numpy as np


class MaxPooling:
    """
    Max Pooling of input
    """

    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None
        self.dx = None

    def forward(self, x):
        """
        Forward pass of max pooling
        :param x: input, (N, C, H, W)
        :return: The output by max pooling with kernel_size and stride
        """
        out = None
        #############################################################################
        # TODO: Implement the max pooling forward pass.                             #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #############################################################################
        N, C, H, W = x.shape
        H_out = (H - self.kernel_size) // self.stride + 1
        W_out = (W - self.kernel_size) // self.stride + 1
        out = np.zeros((N, C, H_out, W_out))

        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        h1 = h * self.stride
                        h2 = h1 + self.kernel_size
                        w1 = w * self.stride
                        w2 = w1 + self.kernel_size
                        out[n, c, h, w] = np.max(x[n, c, h1:h2, w1:w2])


        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = (x, H_out, W_out)
        return out

    def backward(self, dout):
        """
        Backward pass of max pooling
        :param dout: Upstream derivatives
        :return:
        """
        x, H_out, W_out = self.cache
        #############################################################################
        # TODO: Implement the max pooling backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                     #
        #       2) You may find np.unravel_index useful                             #
        #############################################################################
        N, C, H, W = x.shape
        self.dx = np.zeros_like(x)

        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        h1 = h * self.stride
                        h2 = h1 + self.kernel_size
                        w1 = w * self.stride
                        w2 = w1 + self.kernel_size
                        window = x[n, c, h1:h2, w1:w2]
                        max_index = np.argmax(window)
                        max_coord = np.unravel_index(max_index, window.shape)
                        self.dx[n, c, h1 + max_coord[0], w1 + max_coord[1]] = dout[n, c, h, w]

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
