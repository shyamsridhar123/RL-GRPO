"""
Utility functions for RL training
"""

import torch
import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Setup logging configuration."""
    log_level = config.get('level', 'INFO')
    log_format = config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=[
            logging.FileHandler(f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def save_checkpoint(model: torch.nn.Module, 
                   episode: int, 
                   path: str,
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   additional_data: Optional[Dict[str, Any]] = None) -> None:
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'episode': episode,
        'timestamp': datetime.now().isoformat()
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if additional_data is not None:
        checkpoint.update(additional_data)
    
    torch.save(checkpoint, path)


def load_checkpoint(path: str) -> Dict[str, Any]:
    """Load model checkpoint."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    checkpoint = torch.load(path, map_location='cpu')
    return checkpoint


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """Check if training should stop early."""
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop


class MovingAverage:
    """Moving average tracker."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.values = []
    
    def update(self, value: float) -> float:
        """Update with new value and return current average."""
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)
        return sum(self.values) / len(self.values)
    
    def get_average(self) -> float:
        """Get current moving average."""
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)


def calculate_gae(rewards: torch.Tensor, 
                  values: torch.Tensor, 
                  next_values: torch.Tensor,
                  gamma: float = 0.99, 
                  lambda_: float = 0.95) -> torch.Tensor:
    """Calculate Generalized Advantage Estimation (GAE)."""
    deltas = rewards + gamma * next_values - values
    advantages = torch.zeros_like(rewards)
    
    advantage = 0
    for t in reversed(range(len(rewards))):
        advantage = deltas[t] + gamma * lambda_ * advantage
        advantages[t] = advantage
    
    return advantages


def normalize_advantages(advantages: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize advantages to have zero mean and unit variance."""
    return (advantages - advantages.mean()) / (advantages.std() + eps)
