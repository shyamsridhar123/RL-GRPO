"""
Evaluation script for trained RL agents
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import json
from typing import Dict, Any, List
from tqdm import tqdm

# Local imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents import PPOAgent
from src.models import RLLanguageModel, ActorCriticModel
from src.environments import TextGenerationEnv
from src.utils import load_checkpoint, setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate trained RL agent")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--config_path", type=str, required=True,
                       help="Path to training config file")
    parser.add_argument("--num_episodes", type=int, default=10,
                       help="Number of evaluation episodes")
    parser.add_argument("--prompts_file", type=str, default=None,
                       help="File containing evaluation prompts")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json",
                       help="Output file for results")
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_prompts(prompts_file: str) -> List[str]:
    """Load evaluation prompts from file."""
    if prompts_file and os.path.exists(prompts_file):
        with open(prompts_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        # Default prompts
        prompts = [
            "The quick brown fox",
            "Once upon a time",
            "In a galaxy far, far away",
            "The future of artificial intelligence",
            "Climate change is"
        ]
    return prompts


def evaluate_episode(agent, env, prompt: str, max_steps: int = 100) -> Dict[str, Any]:
    """Evaluate one episode."""
    # Reset environment with prompt
    state, info = env.reset(options={'prompt': prompt})
    state_tensor = torch.tensor(state, dtype=torch.long).unsqueeze(0)
    
    generated_text = prompt
    episode_reward = 0
    episode_length = 0
    
    for step in range(max_steps):
        # Select action
        with torch.no_grad():
            action = agent.select_action(state_tensor)
        
        # Take step in environment
        next_state, reward, terminated, truncated, info = env.step(action.item())
        
        # Update metrics
        episode_reward += reward
        episode_length += 1
        generated_text = info.get('text', generated_text)
        
        # Check if episode is done
        if terminated or truncated:
            break
        
        # Update state
        state = next_state
        state_tensor = torch.tensor(state, dtype=torch.long).unsqueeze(0)
    
    return {
        'prompt': prompt,
        'generated_text': generated_text,
        'episode_reward': episode_reward,
        'episode_length': episode_length,
        'completed': terminated
    }


def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    rewards = [r['episode_reward'] for r in results]
    lengths = [r['episode_length'] for r in results]
    completion_rate = sum(1 for r in results if r['completed']) / len(results)
    
    metrics = {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'completion_rate': completion_rate,
        'min_reward': np.min(rewards),
        'max_reward': np.max(rewards)
    }
    
    return metrics


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Load config
    config = load_config(args.config_path)
    
    # Setup logging
    logger = setup_logging(config.get('logging', {}))
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create model
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
    
    model.to(device)
    
    # Load checkpoint
    checkpoint = load_checkpoint(args.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"Loaded model from {args.model_path}")
    
    # Create environment
    env_config = config['environment']
    env = TextGenerationEnv(
        tokenizer_name=env_config.get('tokenizer_name', 'gpt2'),
        max_length=env_config.get('max_length', 100)
    )
    
    # Create agent
    agent = PPOAgent(model, config['agent'])
    
    # Load evaluation prompts
    prompts = load_prompts(args.prompts_file)
    logger.info(f"Loaded {len(prompts)} evaluation prompts")
    
    # Run evaluation
    results = []
    for i in tqdm(range(args.num_episodes), desc="Evaluating"):
        prompt = prompts[i % len(prompts)]  # Cycle through prompts
        result = evaluate_episode(agent, env, prompt)
        results.append(result)
        
        # Log sample results
        if i < 5:  # Show first 5 results
            logger.info(f"Sample {i+1}:")
            logger.info(f"  Prompt: {result['prompt']}")
            logger.info(f"  Generated: {result['generated_text']}")
            logger.info(f"  Reward: {result['episode_reward']:.2f}")
            logger.info(f"  Length: {result['episode_length']}")
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Log metrics
    logger.info("Evaluation Results:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    # Save results
    output_data = {
        'metrics': metrics,
        'results': results,
        'config': config,
        'model_path': args.model_path,
        'num_episodes': args.num_episodes
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
