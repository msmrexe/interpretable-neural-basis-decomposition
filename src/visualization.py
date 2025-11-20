import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

def plot_training_curves(history_dict, title="Optimization Dynamics"):
    """
    Plots loss curves for multiple optimizers or experiments side-by-side.
    
    Args:
        history_dict (dict): { 'OptimizerName': [loss_epoch_1, loss_epoch_2, ...] }
        title (str): Plot title.
    """
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    for name, losses in history_dict.items():
        epochs = range(1, len(losses) + 1)
        plt.plot(epochs, losses, label=f"{name}", linewidth=2)
        
    plt.title(title, fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_basis_mechanisms(x_vals, basis_activations, predicted_output, ground_truth=None):
    """
    Visualizes the 'Mechanistic' view of the network: 
    How individual ReLU neurons sum up to create the output function.
    
    Args:
        x_vals (np.array): Input domain values (N,).
        basis_activations (np.array): The weighted activations of hidden neurons (N, num_neurons).
                                      Formula: w_out_i * ReLU(w_in_i * x + b_i)
        predicted_output (np.array): The sum of basis_activations (N,).
        ground_truth (np.array, optional): The true function values (N,).
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # 1. Basis Functions (The Engine Room)
    num_neurons = basis_activations.shape[1]
    # Use a colormap to distinguish neurons, but keep alpha low for "ghost" effect
    colors = plt.cm.jet(np.linspace(0, 1, num_neurons))
    
    for i in range(num_neurons):
        # Plot individual basis functions (weighted ReLUs)
        axes[0].plot(x_vals, basis_activations[:, i], color=colors[i], alpha=0.4, linewidth=1)
        
    axes[0].set_title(f"Internal Representation: {num_neurons} Weighted Basis Functions", fontsize=14)
    axes[0].set_ylabel("Neuron Contribution", fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # 2. Function Approximation (The Output)
    if ground_truth is not None:
        axes[1].plot(x_vals, ground_truth, label='Ground Truth (Target)', color='black', linestyle='--', linewidth=2.5)
        
    axes[1].plot(x_vals, predicted_output, label='Model Output (Sum of Bases)', color='crimson', linewidth=2.5)
    
    axes[1].set_title("Universal Approximation: Superposition of Bases", fontsize=14)
    axes[1].set_xlabel("Input (x)", fontsize=12)
    axes[1].set_ylabel("Output (y)", fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_decision_boundary(model, X, y, resolution=500, margin=0.5, device='cpu'):
    """
    Plots the 2D decision boundary for the Spiral dataset.
    
    Args:
        model (torch.nn.Module): Trained PyTorch model.
        X (torch.Tensor or np.array): Data points.
        y (torch.Tensor or np.array): Labels.
        resolution (int): Grid density.
        margin (float): Plot margin around data min/max.
    """
    # Convert to numpy for plotting
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
        
    # Create Grid
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    
    # Flatten for prediction
    grid_tensor = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]).to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        logits = model(grid_tensor)
        predictions = torch.argmax(logits, dim=1)
        
    predictions = predictions.cpu().numpy().reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(8, 8))
    plt.contourf(xx, yy, predictions, alpha=0.7, cmap='jet')
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k', cmap='jet', alpha=0.8)
    
    plt.title("Geometric Topology: Learned Decision Boundary", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
