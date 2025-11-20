from .layers import Layer

class NumpyMLP:
    """
    Modular Multi-Layer Perceptron container.
    Manages a sequence of layers and activation functions.
    """
    def __init__(self):
        self.layers = []
        self.is_training = True

    def add(self, layer):
        """Adds a layer (or activation) to the network."""
        self.layers.append(layer)

    def forward(self, x):
        """Passes input through all layers sequentially."""
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, grad_output):
        """
        Backpropagates gradients through all layers in reverse order.
        Args:
            grad_output: Gradient w.r.t the output of the network (dL/dY_hat).
        """
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def parameters(self):
        """
        Retrieves all trainable parameters from the network.
        Used by the Optimizer to update weights.
        
        Returns:
            List of parameter dictionaries (references to the actual arrays).
        """
        params = []
        for layer in self.layers:
            if hasattr(layer, 'get_params'):
                params.extend(layer.get_params())
        return params
    
    def train(self):
        self.is_training = True
        
    def eval(self):
        self.is_training = False

    def __repr__(self):
        return f"NumpyMLP(\n  " + "\n  ".join([str(l) for l in self.layers]) + "\n)"
