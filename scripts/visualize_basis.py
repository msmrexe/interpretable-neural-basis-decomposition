import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils import setup_logger, set_seed, get_base_args
from src.data_loader import generate_sine_wave
from src.visualization import plot_basis_mechanisms
from src.torch_engine import ExplainableReLUNet

def run_visualization(args):
    logger = setup_logger("BasisViz")
    set_seed(args.seed)
    
    # 1. Generate Data (Sine Wave)
    # We want a clean sine wave for visualization
    x_np, y_np = generate_sine_wave(n_samples=200, noise=0.0)
    
    # Convert to PyTorch tensors
    x_train = torch.FloatTensor(x_np)
    y_train = torch.FloatTensor(y_np).reshape(-1, 1)
    
    # 2. Setup Model
    # Hidden dim = number of basis functions (ReLU "kinks")
    model = ExplainableReLUNet(hidden_dim=args.hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.MSELoss()
    
    logger.info(f"Training ExplainableReLUNet with {args.hidden_dim} basis functions...")
    
    # 3. Train
    loss_history = []
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if (epoch + 1) % 100 == 0:
            logger.info(f"Epoch {epoch+1}/{args.epochs} - MSE Loss: {loss.item():.5f}")

    # 4. Extract Basis Decomposition
    model.eval()
    with torch.no_grad():
        # We use a dense grid for plotting smooth basis functions
        x_plot_np = np.linspace(0, 2*np.pi, 500).reshape(-1, 1)
        x_plot = torch.FloatTensor(x_plot_np)
        
        final_pred, contributions = model.forward_decomposed(x_plot)
        
        # Convert back to numpy
        final_pred = final_pred.numpy()
        contributions = contributions.numpy()
        ground_truth = np.sin(x_plot_np)

    # 5. Visualize
    logger.info("Generating Basis Decomposition Plot...")
    plot_basis_mechanisms(x_plot_np, contributions, final_pred, ground_truth)
    
    # Save
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/basis_decomposition.png")
    logger.info("Plot saved to outputs/basis_decomposition.png")

if __name__ == "__main__":
    parser = get_base_args()
    parser.add_argument('--hidden_dim', type=int, default=10, help="Number of ReLU basis functions")
    parser.add_argument('--epochs', type=int, default=1000, help="Training epochs")
    
    args = parser.parse_args()
    run_visualization(args)
