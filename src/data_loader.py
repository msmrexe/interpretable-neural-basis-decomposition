import numpy as np
import torch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

class PolynomialFeatureExpander:
    """
    Expands 2D input features into polynomial features up to degree P.
    Formula: x1^n * x2^m where n + m <= P.
    """
    def __init__(self, degree):
        self.degree = degree
        # Calculate number of features: (P+1)(P+2)/2 - 1 (excluding bias)
        self.num_features = (degree + 1) * (degree + 2) // 2 - 1

    def transform(self, X):
        """
        Args:
            X (np.ndarray or torch.Tensor): Input data of shape (N, 2).
        Returns:
            np.ndarray or torch.Tensor: Expanded data of shape (N, num_features).
        """
        is_torch = isinstance(X, torch.Tensor)
        if is_torch:
            X = X.detach().cpu().numpy()
            
        N = X.shape[0]
        poly_X = np.zeros((N, self.num_features))
        
        count = 0
        # Generate features order: x^1 y^0, x^0 y^1, x^2 y^0, x^1 y^1, ...
        for i in range(self.degree + 1):
            for j in range(self.degree + 1):
                # Exclude the constant term (bias is handled by the layer)
                # Exclude terms where combined degree exceeds max degree
                if 0 < i + j <= self.degree:
                    term = (X[:, 0] ** i) * (X[:, 1] ** j)
                    poly_X[:, count] = term
                    count += 1
                    
        if is_torch:
            return torch.FloatTensor(poly_X)
        return poly_X

def generate_sine_wave(n_samples=100, noise=0.0):
    """
    Generates 1D Sine wave data for function approximation tasks.
    Target: y = sin(x)
    Range: [0, 2*pi]
    """
    X = np.linspace(0, 2 * np.pi, n_samples).reshape(-1, 1)
    y = np.sin(X)
    
    if noise > 0:
        y += np.random.normal(0, noise, y.shape)
        
    return X.astype(np.float32), y.astype(np.float32)

def generate_spiral_data(n_points=1000, K=4, sigma=0.4):
    """
    Generates the K-class Spiral dataset used for Depth vs. Width analysis.
    
    Formula:
    X_k(t) = t * [sin(2pi/K * (2t + k - 1)) + N(0, sigma), 
                  cos(2pi/K * (2t + k - 1)) + N(0, sigma)]
    
    Args:
        n_points (int): Total number of points.
        K (int): Number of spiral arms (classes).
        sigma (float): Noise magnitude.
        
    Returns:
        X (np.ndarray): (N, 2) coordinates.
        y (np.ndarray): (N,) labels.
    """
    N_per_class = n_points // K
    X = np.zeros((N_per_class * K, 2))
    y = np.zeros(N_per_class * K)
    
    for k in range(K):
        t = np.linspace(0, 1, N_per_class)
        
        # The spiral arm calculation
        # Note: We use random noise scaled by sigma^2 as per original math spec
        noise_x = np.random.randn(N_per_class) * (sigma ** 2)
        noise_y = np.random.randn(N_per_class) * (sigma ** 2)
        
        theta = (2 * np.pi / K) * (2 * t + k) 
        
        # Fill x1 coordinate
        X[k*N_per_class : (k+1)*N_per_class, 0] = t * (np.sin(theta) + noise_x)
        # Fill x2 coordinate
        X[k*N_per_class : (k+1)*N_per_class, 1] = t * (np.cos(theta) + noise_y)
        # Fill labels
        y[k*N_per_class : (k+1)*N_per_class] = k
        
    return X.astype(np.float32), y.astype(np.int64)

def load_mnist(train_size=60000, test_size=10000):
    """
    Fetches MNIST, normalizes to [0,1], and performs One-Hot Encoding.
    
    Returns:
        x_train, y_train, x_test, y_test (np.ndarrays)
    """
    print("[Data] Downloading/Loading MNIST from OpenML...")
    # fetch_openml caches data in ~/scikit_learn_data by default
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    
    x = mnist["data"].astype('float32')
    y = mnist["target"].astype('int')
    
    # Normalize
    x /= 255.0
    
    # One-hot encode
    num_labels = len(np.unique(y))
    y_encoded = to_categorical(y, num_labels)
    
    # Split
    # Note: MNIST standard split is 60k train / 10k test. 
    # We verify sizes to prevent index errors if custom sizes requested.
    if train_size + test_size > 70000:
        raise ValueError("Requested size exceeds MNIST dataset size (70k)")
        
    x_train, x_test, y_train, y_test = train_test_split(
        x, y_encoded, 
        train_size=train_size, 
        test_size=test_size, 
        stratify=y, # Ensure balanced classes
        random_state=42
    )
    
    print(f"[Data] MNIST Loaded. Train: {x_train.shape}, Test: {x_test.shape}")
    return x_train, y_train, x_test, y_test
