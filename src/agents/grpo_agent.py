"""
GRPO (Group Relative Policy Optimization) Agent Implementation
More efficient than PPO for CPU training and memory-constrained environments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
import numpy as np


class GRPOAgent:
    """
    Group Relative Policy Optimization (GRPO) agent.
    
    GRPO is more efficient than PPO for CPU training because:
    1. No value function estimation required
    2. Uses relative rewards within batches
    3. Simpler gradient computation
    4. Lower memory footprint
    5. More stable convergence
    """
    
    def __init__(self, 
                 model: nn.Module,
                 config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.device = torch.device("cpu")  # Optimized for CPU
        self.model.to(self.device)
        
        # GRPO specific parameters
        self.lr = config.get('learning_rate', 1e-4)
        self.gamma = config.get('gamma', 0.99)
        self.batch_size = config.get('batch_size', 32)
        self.relative_clip = config.get('relative_clip', 0.2)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Memory efficient experience storage
        self.experience_buffer = []
        
    def select_action(self, 
                     input_ids: torch.Tensor,
                     attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Select action using current policy."""
        
        with torch.no_grad():
            # Get policy logits
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            if isinstance(outputs, dict):
                logits = outputs['policy_logits'][:, -1, :]  # Last token
            else:
                logits = outputs[:, -1, :]
            
            # Sample action
            probs = F.softmax(logits, dim=-1)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
        return {
            'action': action,
            'log_prob': log_prob,
            'entropy': action_dist.entropy()
        }
    
    def store_experience(self, 
                        input_ids: torch.Tensor,
                        action: torch.Tensor,
                        reward: float,
                        log_prob: torch.Tensor,
                        attention_mask: Optional[torch.Tensor] = None):
        """Store experience in buffer."""
        
        experience = {
            'input_ids': input_ids.cpu(),
            'action': action.cpu(),
            'reward': reward,
            'log_prob': log_prob.cpu(),
            'attention_mask': attention_mask.cpu() if attention_mask is not None else None
        }
        
        self.experience_buffer.append(experience)
    
    def compute_relative_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute relative rewards - key innovation of GRPO.
        
        Instead of using absolute rewards or advantages (like PPO),
        GRPO uses rewards relative to the batch average.
        """
        
        # Method 1: Simple mean subtraction
        if self.config.get('relative_method', 'mean') == 'mean':
            mean_reward = rewards.mean()
            return rewards - mean_reward
        
        # Method 2: Rank-based (more stable)
        elif self.config.get('relative_method') == 'rank':
            # Convert to ranks, then normalize
            sorted_indices = torch.argsort(rewards)
            ranks = torch.zeros_like(rewards)
            ranks[sorted_indices] = torch.arange(len(rewards), dtype=rewards.dtype)
            ranks = (ranks - ranks.mean()) / (ranks.std() + 1e-8)
            return ranks
        
        # Method 3: Percentile-based
        else:  # 'percentile'
            median_reward = torch.median(rewards)
            return rewards - median_reward
    
    def update(self) -> Dict[str, float]:
        """
        Update agent using GRPO algorithm.
        
        GRPO is more efficient than PPO because:
        - No value function to train
        - No GAE computation
        - Simpler policy gradient with relative rewards
        """
        
        if len(self.experience_buffer) < self.batch_size:
            return {'loss': 0.0, 'samples': len(self.experience_buffer)}
        
        # Sample batch from buffer
        batch_indices = np.random.choice(
            len(self.experience_buffer), 
            size=min(self.batch_size, len(self.experience_buffer)),
            replace=False
        )
        
        batch_experiences = [self.experience_buffer[i] for i in batch_indices]
        
        # Prepare batch data
        input_ids = torch.stack([exp['input_ids'] for exp in batch_experiences]).to(self.device)
        actions = torch.stack([exp['action'] for exp in batch_experiences]).to(self.device)
        rewards = torch.tensor([exp['reward'] for exp in batch_experiences], 
                              dtype=torch.float32, device=self.device)
        old_log_probs = torch.stack([exp['log_prob'] for exp in batch_experiences]).to(self.device)
        
        # Handle attention masks
        if batch_experiences[0]['attention_mask'] is not None:
            attention_mask = torch.stack([exp['attention_mask'] for exp in batch_experiences]).to(self.device)
        else:
            attention_mask = None
        
        # Compute relative rewards (key GRPO innovation)
        relative_rewards = self.compute_relative_rewards(rewards)
        
        # Get current policy
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        if isinstance(outputs, dict):
            current_logits = outputs['policy_logits'][:, -1, :]
        else:
            current_logits = outputs[:, -1, :]
        
        current_probs = F.softmax(current_logits, dim=-1)
        current_log_probs = torch.log(current_probs.gather(1, actions.unsqueeze(1)).squeeze(1) + 1e-8)
        
        # GRPO policy loss (simpler than PPO)
        # No clipping needed due to relative reward formulation
        policy_ratio = torch.exp(current_log_probs - old_log_probs)
        
        # Optional: Add stability clipping (less aggressive than PPO)
        if self.relative_clip > 0:
            policy_ratio = torch.clamp(policy_ratio, 1 - self.relative_clip, 1 + self.relative_clip)
        
        policy_loss = -(policy_ratio * relative_rewards).mean()
        
        # Entropy bonus for exploration
        entropy_bonus = -(current_probs * torch.log(current_probs + 1e-8)).sum(dim=-1).mean()
        entropy_coef = self.config.get('entropy_coef', 0.01)
        
        total_loss = policy_loss - entropy_coef * entropy_bonus
        
        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        if self.config.get('grad_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
        
        self.optimizer.step()
        
        # Clear buffer (or implement sliding window)
        if self.config.get('clear_buffer_after_update', True):
            self.experience_buffer.clear()
        else:
            # Keep only recent experiences
            max_buffer_size = self.config.get('max_buffer_size', 1000)
            if len(self.experience_buffer) > max_buffer_size:
                self.experience_buffer = self.experience_buffer[-max_buffer_size:]
        
        return {
            'loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'entropy': entropy_bonus.item(),
            'mean_reward': rewards.mean().item(),
            'relative_reward_std': relative_rewards.std().item(),
            'samples': len(batch_experiences)
        }
    
    def save(self, path: str) -> None:
        """Save GRPO agent."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'experience_buffer': self.experience_buffer
        }, path)
    
    def load(self, path: str) -> None:
        """Load GRPO agent."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.experience_buffer = checkpoint.get('experience_buffer', [])


class AdaptiveGRPOAgent(GRPOAgent):
    """
    Adaptive GRPO that automatically adjusts learning rate and 
    relative reward computation based on training progress.
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        super().__init__(model, config)
        
        self.initial_lr = self.lr
        self.reward_history = []
        self.update_count = 0
        
    def update(self) -> Dict[str, float]:
        """Update with adaptive learning rate."""
        
        # Regular GRPO update
        update_info = super().update()
        
        if update_info['samples'] > 0:
            self.update_count += 1
            self.reward_history.append(update_info['mean_reward'])
            
            # Keep only recent history
            if len(self.reward_history) > 100:
                self.reward_history = self.reward_history[-100:]
            
            # Adaptive learning rate
            if len(self.reward_history) > 10:
                recent_trend = np.mean(self.reward_history[-10:]) - np.mean(self.reward_history[-20:-10])
                
                if recent_trend > 0:  # Improving
                    self.lr = min(self.initial_lr * 1.1, self.initial_lr * 2.0)
                else:  # Not improving
                    self.lr = max(self.initial_lr * 0.9, self.initial_lr * 0.5)
                
                # Update optimizer learning rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr
                
                update_info['current_lr'] = self.lr
        
        return update_info
