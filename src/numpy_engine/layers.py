import numpy as np

class Layer:
    """Base class for all layers."""
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, grad_output):
        raise NotImplementedError
    
    def get_params(self):
        """Returns parameters and gradients for the optimizer."""
        return []

class Linear(Layer):
    """
    Fully Connected (Dense) Layer.
    Implements Y = XW + b
    
    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
    """
    def __init__(self, in_features, out_features):
        # Xavier/Glorot Initialization
        limit = np.sqrt(6 / (in_features + out_features))
        self.W = np.random.uniform(-limit, limit, (in_features, out_features))
        self.b = np.zeros((1, out_features))
        
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        
        self.input_cache = None

    def forward(self, x):
        """
        Args:
            x: Input data of shape (Batch_Size, in_features)
        Returns:
            Linear transform of shape (Batch_Size, out_features)
        """
        self.input_cache = x
        return np.dot(x, self.W) + self.b

    def backward(self, grad_output):
        """
        Args:
            grad_output: Gradient flowing from the next layer (Batch_Size, out_features)
        Returns:
            grad_input: Gradient to flow to the previous layer (Batch_Size, in_features)
        """
        # dL/dW = X.T * dL/dY
        self.dW = np.dot(self.input_cache.T, grad_output)
        
        # dL/db = sum(dL/dY) across batch
        self.db = np.sum(grad_output, axis=0, keepdims=True)
        
        # dL/dX = dL/dY * W.T
        grad_input = np.dot(grad_output, self.W.T)
        return grad_input

    def get_params(self):
        """
        Returns a list of parameter dictionaries for the optimizer.
        References mutable objects (arrays) so optimizer updates persist.
        """
        return [
            {'name': 'W', 'data': self.W, 'grad': self.dW},
            {'name': 'b', 'data': self.b, 'grad': self.db}
        ]
    
    def __str__(self):
        return f"Linear({self.W.shape[0]} -> {self.W.shape[1]})"
