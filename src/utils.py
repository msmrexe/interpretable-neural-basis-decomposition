import logging
import random
import numpy as np
import torch
import os
import argparse

def setup_logger(name="NeuralBasis", log_file="project.log", level=logging.INFO):
    """
    Sets up a logger that outputs to both console and a file.
    
    Args:
        name (str): Name of the logger.
        log_file (str): Path to the log file.
        level (int): Logging level (e.g., logging.INFO).
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Check if handlers already exist to avoid duplicate logs
    if not logger.handlers:
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File Handler
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    
    Args:
        seed (int): The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # If using the os environment to seed hash randomization
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"[System] Random seed set to {seed}")

def get_base_args():
    """
    Returns a basic argument parser with common flags used across scripts.
    Scripts can add to this parser.
    """
    parser = argparse.ArgumentParser(description="Interpretable Neural Basis Decomposition")
    parser.add_argument('--seed', type=int, default=2024, help='Random seed for reproducibility')
    parser.add_argument('--log_file', type=str, default='experiment.log', help='Path to log file')
    return parser
