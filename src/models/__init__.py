"""
LLM models for RL training
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Dict, Any, Optional, Tuple


class RLLanguageModel(nn.Module):
    """
    Language model wrapper for RL training.
    """
    
    def __init__(self, 
                 model_name: str = "gpt2",
                 freeze_base: bool = False,
                 add_value_head: bool = True):
        super().__init__()
        
        # Load pretrained model and tokenizer
        self.config = AutoConfig.from_pretrained(model_name)
        self.base_model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Freeze base model if requested
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        # Add policy head (for action selection)
        self.policy_head = nn.Linear(self.config.hidden_size, self.config.vocab_size)
        
        # Add value head (for critic)
        if add_value_head:
            self.value_head = nn.Linear(self.config.hidden_size, 1)
        else:
            self.value_head = None
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                return_dict: bool = True) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        
        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get last hidden state
        last_hidden_state = outputs.last_hidden_state
        
        # Policy logits
        policy_logits = self.policy_head(last_hidden_state)
        
        # Value estimates
        values = None
        if self.value_head is not None:
            values = self.value_head(last_hidden_state).squeeze(-1)
        
        if return_dict:
            return {
                'policy_logits': policy_logits,
                'values': values,
                'hidden_states': last_hidden_state
            }
        else:
            return policy_logits, values
    
    def generate_action_probs(self, 
                            input_ids: torch.Tensor,
                            attention_mask: Optional[torch.Tensor] = None,
                            temperature: float = 1.0) -> torch.Tensor:
        """Generate action probabilities for the last token."""
        
        outputs = self.forward(input_ids, attention_mask)
        logits = outputs['policy_logits'][:, -1, :]  # Get last token logits
        
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature
        
        # Apply softmax
        action_probs = torch.softmax(logits, dim=-1)
        
        return action_probs
    
    def get_value(self, 
                  input_ids: torch.Tensor,
                  attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get value estimates."""
        
        if self.value_head is None:
            raise ValueError("Value head not available")
        
        outputs = self.forward(input_ids, attention_mask)
        return outputs['values'][:, -1]  # Return value for last token


class ActorCriticModel(nn.Module):
    """
    Actor-Critic model for RL training.
    """
    
    def __init__(self, 
                 base_model_name: str = "gpt2",
                 hidden_size: int = 768,
                 freeze_base: bool = False):
        super().__init__()
        
        # Shared base model
        self.base_model = RLLanguageModel(
            model_name=base_model_name,
            freeze_base=freeze_base,
            add_value_head=False
        )
        
        # Actor (policy) network
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.base_model.config.vocab_size)
        )
        
        # Critic (value) network
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning policy logits and values."""
        
        # Get base model features
        outputs = self.base_model.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        features = outputs.last_hidden_state[:, -1, :]  # Use last token
        
        # Get policy logits and values
        policy_logits = self.actor(features)
        values = self.critic(features).squeeze(-1)
        
        return policy_logits, values
    
    def get_action_and_value(self, 
                           input_ids: torch.Tensor,
                           attention_mask: Optional[torch.Tensor] = None,
                           action: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log probability, and value."""
        
        policy_logits, values = self.forward(input_ids, attention_mask)
        
        # Create action distribution
        action_dist = torch.distributions.Categorical(logits=policy_logits)
        
        if action is None:
            action = action_dist.sample()
        
        log_prob = action_dist.log_prob(action)
        
        return action, log_prob, values
