"""
Training script for RL agents
"""

import torch
import torch.nn as nn
import numpy as np
import yaml
import argparse
import os
from datetime import datetime
from typing import Dict, Any, List
import wandb
from tqdm import tqdm

# Local imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import PPOAgent
from models import RLLanguageModel, ActorCriticModel
from environments import TextGenerationEnv
from utils import setup_logging, save_checkpoint, load_checkpoint


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train RL agent for LLM")
    parser.add_argument("--config", type=str, required=True, 
                       help="Path to config file")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--wandb", action="store_true",
                       help="Use Weights & Biases logging")
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: Dict[str, Any]) -> nn.Module:
    """Create model based on configuration."""
    model_config = config['model']
    model_type = model_config.get('type', 'rl_language_model')
    
    if model_type == 'rl_language_model':
        model = RLLanguageModel(
            model_name=model_config.get('name', 'gpt2'),
            freeze_base=model_config.get('freeze_base', False),
            add_value_head=model_config.get('add_value_head', True)
        )
    elif model_type == 'actor_critic':
        model = ActorCriticModel(
            base_model_name=model_config.get('name', 'gpt2'),
            hidden_size=model_config.get('hidden_size', 768),
            freeze_base=model_config.get('freeze_base', False)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def create_environment(config: Dict[str, Any]) -> TextGenerationEnv:
    """Create environment based on configuration."""
    env_config = config['environment']
    env_type = env_config.get('type', 'text_generation')
    
    if env_type == 'text_generation':
        env = TextGenerationEnv(
            tokenizer_name=env_config.get('tokenizer_name', 'gpt2'),
            max_length=env_config.get('max_length', 100)
        )
    else:
        raise ValueError(f"Unknown environment type: {env_type}")
    
    return env


def train_episode(agent, env, config: Dict[str, Any]) -> Dict[str, float]:
    """Train one episode."""
    episode_data = {
        'states': [],
        'actions': [],
        'rewards': [],
        'log_probs': [],
        'values': []
    }
    
    # Reset environment
    state, info = env.reset()
    state_tensor = torch.tensor(state, dtype=torch.long).unsqueeze(0)
    
    episode_reward = 0
    episode_length = 0
    
    for step in range(config['training']['max_steps_per_episode']):
        # Select action
        action = agent.select_action(state_tensor)
        
        # Take step in environment
        next_state, reward, terminated, truncated, info = env.step(action.item())
        
        # Store experience
        episode_data['states'].append(state_tensor)
        episode_data['actions'].append(action)
        episode_data['rewards'].append(reward)
        
        # Update metrics
        episode_reward += reward
        episode_length += 1
        
        # Check if episode is done
        if terminated or truncated:
            break
        
        # Update state
        state = next_state
        state_tensor = torch.tensor(state, dtype=torch.long).unsqueeze(0)
    
    # Convert to tensors
    episode_data['states'] = torch.cat(episode_data['states'])
    episode_data['actions'] = torch.cat(episode_data['actions'])
    episode_data['rewards'] = torch.tensor(episode_data['rewards'], dtype=torch.float32)
    
    # Update agent
    loss_info = agent.update(episode_data)
    
    return {
        'episode_reward': episode_reward,
        'episode_length': episode_length,
        'final_text': info.get('text', ''),
        **loss_info
    }


def main():
    """Main training function."""
    args = parse_args()
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging(config.get('logging', {}))
    
    # Initialize wandb if requested
    if args.wandb:
        wandb.init(
            project=config.get('project_name', 'llm-rl'),
            config=config,
            name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create model
    model = create_model(config)
    model.to(device)
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create environment
    env = create_environment(config)
    logger.info(f"Created environment: {type(env).__name__}")
    
    # Create agent
    agent = PPOAgent(model, config['agent'])
    logger.info("Created PPO agent")
    
    # Resume from checkpoint if requested
    start_episode = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_episode = checkpoint['episode']
        logger.info(f"Resumed from episode {start_episode}")
    
    # Training loop
    num_episodes = config['training']['num_episodes']
    save_interval = config['training'].get('save_interval', 100)
    
    for episode in tqdm(range(start_episode, num_episodes), desc="Training"):
        # Train episode
        episode_info = train_episode(agent, env, config)
        
        # Log metrics
        if args.wandb:
            wandb.log({
                'episode': episode,
                'episode_reward': episode_info['episode_reward'],
                'episode_length': episode_info['episode_length'],
                'loss': episode_info.get('loss', 0)
            })
        
        # Print progress
        if episode % 10 == 0:
            logger.info(f"Episode {episode}: Reward={episode_info['episode_reward']:.2f}, "
                       f"Length={episode_info['episode_length']}, "
                       f"Loss={episode_info.get('loss', 0):.4f}")
            logger.info(f"Generated text: {episode_info['final_text']}")
        
        # Save checkpoint
        if episode % save_interval == 0:
            checkpoint_path = f"checkpoints/checkpoint_episode_{episode}.pt"
            save_checkpoint(model, episode, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    # Final save
    final_path = "checkpoints/final_model.pt"
    save_checkpoint(model, num_episodes, final_path)
    logger.info(f"Training completed. Final model saved: {final_path}")


if __name__ == "__main__":
    main()
