"""
Text generation environment for RL training
"""

import gymnasium as gym
import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional
from transformers import AutoTokenizer


class TextGenerationEnv(gym.Env):
    """
    Custom environment for text generation using RL.
    """
    
    def __init__(self, 
                 tokenizer_name: str = "gpt2",
                 max_length: int = 100,
                 reward_function: Optional[callable] = None):
        super().__init__()
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.max_length = max_length
        self.vocab_size = self.tokenizer.vocab_size
        
        # Action space: vocabulary size
        self.action_space = gym.spaces.Discrete(self.vocab_size)
        
        # Observation space: sequence of tokens
        self.observation_space = gym.spaces.Box(
            low=0, high=self.vocab_size-1, 
            shape=(self.max_length,), dtype=np.int32
        )
        
        # Reward function
        self.reward_function = reward_function or self._default_reward
        
        # Environment state
        self.current_sequence = []
        self.prompt = ""
        self.step_count = 0
        
    def reset(self, seed=None, options=None):
        """Reset environment."""
        super().reset(seed=seed)
        
        # Initialize with a random prompt or specific prompt from options
        if options and 'prompt' in options:
            self.prompt = options['prompt']
        else:
            self.prompt = "The quick brown fox"
        
        # Tokenize prompt
        prompt_tokens = self.tokenizer.encode(self.prompt, add_special_tokens=False)
        self.current_sequence = prompt_tokens.copy()
        self.step_count = 0
        
        # Create observation
        observation = self._get_observation()
        info = {'prompt': self.prompt, 'tokens': len(self.current_sequence)}
        
        return observation, info
    
    def step(self, action):
        """Take a step in the environment."""
        # Add action (token) to sequence
        self.current_sequence.append(action)
        self.step_count += 1
        
        # Calculate reward
        reward = self.reward_function(self.current_sequence, action)
        
        # Check if episode is done
        terminated = (
            action == self.tokenizer.eos_token_id or 
            len(self.current_sequence) >= self.max_length
        )
        
        truncated = self.step_count >= self.max_length
        
        # Create observation
        observation = self._get_observation()
        
        # Info
        info = {
            'text': self.tokenizer.decode(self.current_sequence),
            'length': len(self.current_sequence),
            'action_token': self.tokenizer.decode([action])
        }
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self):
        """Get current observation."""
        # Pad or truncate sequence to max_length
        obs = np.zeros(self.max_length, dtype=np.int32)
        seq_len = min(len(self.current_sequence), self.max_length)
        obs[:seq_len] = self.current_sequence[-seq_len:]  # Take last max_length tokens
        return obs
    
    def _default_reward(self, sequence: List[int], action: int) -> float:
        """Default reward function - can be customized."""
        # Simple reward: encourage longer sequences, penalize repetition
        text = self.tokenizer.decode(sequence)
        
        # Length reward
        length_reward = 0.1
        
        # Repetition penalty
        if len(sequence) > 1 and action == sequence[-2]:
            repetition_penalty = -0.5
        else:
            repetition_penalty = 0
        
        # End token reward
        end_reward = 1.0 if action == self.tokenizer.eos_token_id else 0
        
        return length_reward + repetition_penalty + end_reward
    
    def render(self):
        """Render current state."""
        text = self.tokenizer.decode(self.current_sequence)
        print(f"Current text: {text}")
        print(f"Length: {len(self.current_sequence)}")


class ConversationEnv(TextGenerationEnv):
    """
    Environment for conversation/dialog generation.
    """
    
    def __init__(self, 
                 tokenizer_name: str = "microsoft/DialoGPT-medium",
                 max_length: int = 200):
        super().__init__(tokenizer_name, max_length)
        
        self.conversation_history = []
    
    def reset(self, seed=None, options=None):
        """Reset conversation environment."""
        self.conversation_history = []
        
        if options and 'context' in options:
            context = options['context']
            self.conversation_history.append(context)
            self.prompt = context
        else:
            self.prompt = "Hello, how are you?"
        
        return super().reset(seed, options)
    
    def _default_reward(self, sequence: List[int], action: int) -> float:
        """Reward function for conversation quality."""
        text = self.tokenizer.decode(sequence)
        
        # Encourage appropriate response length
        length_reward = 0.1 if 5 <= len(sequence) <= 50 else -0.1
        
        # Encourage ending with proper punctuation
        if action in [self.tokenizer.encode('.')[0], 
                     self.tokenizer.encode('!')[0], 
                     self.tokenizer.encode('?')[0]]:
            punctuation_reward = 0.2
        else:
            punctuation_reward = 0
        
        # Discourage repetition
        repetition_penalty = -0.3 if len(sequence) > 2 and action == sequence[-2] else 0
        
        return length_reward + punctuation_reward + repetition_penalty
