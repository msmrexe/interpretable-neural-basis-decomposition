import torch
import torch.nn as nn

class ExplainableReLUNet(nn.Module):
    """
    A specialized 1-Hidden Layer MLP designed for mechanistic visualization.
    It exposes the internal 'basis functions' that sum up to create the output.
    
    Architecture: Input (1) -> Linear -> ReLU -> Linear -> Output (1)
    """
    def __init__(self, hidden_dim=10):
        super().__init__()
        # Layer 1: The "Basis Generators" (Slopes and Kinks)
        self.hidden = nn.Linear(1, hidden_dim)
        # Layer 2: The "Basis Weights" (Heights/Importance)
        self.output = nn.Linear(hidden_dim, 1)
        
        self.activation = nn.ReLU()

    def forward(self, x):
        """Standard forward pass."""
        x = self.hidden(x)
        x = self.activation(x)
        x = self.output(x)
        return x

    def forward_decomposed(self, x):
        """
        Forward pass that returns the individual contribution of each neuron.
        
        Mathematically:
        f(x) = sum( w_out_i * ReLU(w_in_i * x + b_in_i) ) + b_out
        
        Returns:
            y_pred (torch.Tensor): Final prediction (N, 1)
            contributions (torch.Tensor): Weighted activation of each neuron (N, Hidden)
        """
        # 1. Pre-activation: z = w_in * x + b_in
        z = self.hidden(x)
        
        # 2. Activation: a = ReLU(z)
        a = self.activation(z)
        
        # 3. Weighted Contributions: c_i = w_out_i * a_i
        # We multiply element-wise by the output weights.
        # output.weight is shape (1, Hidden), we view it as (1, Hidden) to broadcast
        w_out = self.output.weight  # Shape: (1, Hidden)
        contributions = a * w_out
        
        # 4. Final Sum (plus output bias)
        y_pred = torch.sum(contributions, dim=1, keepdim=True) + self.output.bias
        
        return y_pred, contributions


class SpiralClassifier(nn.Module):
    """
    A flexible Deep MLP for the 2D Depth vs Width experiments.
    """
    def __init__(self, input_dim=2, hidden_dims=[128, 64], output_dim=4):
        super().__init__()
        layers = []
        
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
