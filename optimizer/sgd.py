
import numpy as np
from ._base_optimizer import _BaseOptimizer


class SGD(_BaseOptimizer):
    def __init__(self, model, learning_rate=1e-4, reg=1e-3, momentum=0.9):
        super().__init__(model, learning_rate, reg)
        self.momentum = momentum

    def update(self, model):
        """
        Update model weights based on gradients
        :param model: The model to be updated
        :return: None, but the model weights should be updated
        """
        self.apply_regularization(model)
        
        
            
        for idx, m in enumerate(model.modules):
            if hasattr(m, 'weight'):
                #############################################################################
                # TODO:                                                                     #
                #    1) Momentum updates for weights                                        #
                #############################################################################
                
                # m.weight += self.momentum
                # m.weight -= self.learning_rate * m.dw
                if not hasattr(m,'weight_velocity'):
                    m.weight_velocity = np.zeros_like(m.weight)
                m.weight_velocity = self.momentum * m.weight_velocity - self.learning_rate * m.dw
                m.weight += m.weight_velocity
            
                #############################################################################
                #                              END OF YOUR CODE                             #
                #############################################################################
            if hasattr(m, 'bias'):
                #############################################################################
                # TODO:                                                                     #
                #    1) Momentum updates for bias                                           #
                #############################################################################
                
                # m.bias += self.momentum
                # m.bias -= self.learning_rate * m.db
                
                if not hasattr(m,'bias_velocity'):
                    m.bias_velocity = np.zeros_like(m.bias)
                m.bias_velocity = self.momentum * m.bias_velocity - self.learning_rate * m.db
                m.bias += m.bias_velocity
                #############################################################################
                #                              END OF YOUR CODE                             #
                #############################################################################
                
