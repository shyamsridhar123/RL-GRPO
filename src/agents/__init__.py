"""
Base RL Agent for LLM training
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np

# Import GRPO agent
from .grpo_agent import GRPOAgent, AdaptiveGRPOAgent


class BaseRLAgent(ABC):
    """
    Abstract base class for RL agents used in LLM training.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    @abstractmethod
    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """Select action based on current state."""
        pass
    
    @abstractmethod
    def update(self, experiences: Dict[str, Any]) -> Dict[str, float]:
        """Update agent based on experiences."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save agent model and parameters."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load agent model and parameters."""
        pass


class PPOAgent(BaseRLAgent):
    """
    Proximal Policy Optimization (PPO) agent for LLM training.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 config: Dict[str, Any]):
        super().__init__(model, config)
        
        self.lr = config.get('learning_rate', 3e-4)
        self.gamma = config.get('gamma', 0.99)
        self.eps_clip = config.get('eps_clip', 0.2)
        self.k_epochs = config.get('k_epochs', 4)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """Select action using policy network."""
        with torch.no_grad():
            action_probs = self.model(state)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
        return action
    
    def update(self, experiences: Dict[str, Any]) -> Dict[str, float]:
        """Update PPO agent."""
        states = experiences['states']
        actions = experiences['actions']
        rewards = experiences['rewards']
        old_log_probs = experiences['log_probs']
        
        # Calculate discounted rewards
        discounted_rewards = self._calculate_discounted_rewards(rewards)
        
        # Update for k epochs
        total_loss = 0
        for _ in range(self.k_epochs):
            # Get current policy
            action_probs = self.model(states)
            action_dist = torch.distributions.Categorical(action_probs)
            new_log_probs = action_dist.log_prob(actions)
            
            # Calculate ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Calculate surrogate loss
            surr1 = ratio * discounted_rewards
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * discounted_rewards
            loss = -torch.min(surr1, surr2).mean()
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return {'loss': total_loss / self.k_epochs}
    
    def _calculate_discounted_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """Calculate discounted rewards."""
        discounted = torch.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted[t] = running_add
        return discounted
    
    def save(self, path: str) -> None:
        """Save PPO agent."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)
    
    def load(self, path: str) -> None:
        """Load PPO agent."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
