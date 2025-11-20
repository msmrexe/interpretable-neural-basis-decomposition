import numpy as np

class Loss:
    def forward(self, predictions, targets):
        raise NotImplementedError
    
    def backward(self):
        raise NotImplementedError

class MSELoss(Loss):
    """Mean Squared Error Loss"""
    def forward(self, predictions, targets):
        self.predictions = predictions
        self.targets = targets
        # L = (1/N) * sum((y_pred - y_true)^2)
        return np.mean((predictions - targets) ** 2)
    
    def backward(self):
        # dL/dy_pred = 2 * (y_pred - y_true) / N
        N = self.predictions.shape[0]
        return 2 * (self.predictions - self.targets) / N

class CrossEntropyLoss(Loss):
    """
    Cross Entropy Loss with Softmax handling.
    Expects raw logits if raw_logits=True, otherwise expects probabilities.
    Defaults to raw logits for numerical stability.
    """
    def __init__(self, epsilon=1e-12):
        self.epsilon = epsilon
        self.predictions = None
        self.targets = None
        self.probs = None

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Raw logits (N, C)
            targets: One-hot encoded labels (N, C)
        """
        self.predictions = predictions
        self.targets = targets
        
        # Softmax (Numerically Stable)
        shift_preds = predictions - np.max(predictions, axis=1, keepdims=True)
        exps = np.exp(shift_preds)
        self.probs = exps / np.sum(exps, axis=1, keepdims=True)
        
        # Cross Entropy: -sum(y * log(p))
        # Clip for stability to avoid log(0)
        probs_clipped = np.clip(self.probs, self.epsilon, 1. - self.epsilon)
        loss = -np.sum(targets * np.log(probs_clipped)) / predictions.shape[0]
        return loss
    
    def backward(self):
        """
        Gradient of CrossEntropyLoss with respect to Logits (Z).
        dLoss/dZ = Probabilities - Targets
        """
        N = self.predictions.shape[0]
        return (self.probs - self.targets) / N
