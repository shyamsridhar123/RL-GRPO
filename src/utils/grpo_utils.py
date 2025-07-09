"""
Utilities for GRPO training and evaluation
Includes data processing, reward functions, and evaluation metrics
"""

import re
import json
import torch
import numpy as np
from typing import List, Dict, Any, Callable, Tuple
from datasets import Dataset
import matplotlib.pyplot as plt
from pathlib import Path


class RewardFunctions:
    """Collection of reward functions for different tasks"""
    
    @staticmethod
    def math_reasoning_reward(completions: List[str], **kwargs) -> List[float]:
        """
        Reward function for mathematical reasoning tasks
        Evaluates mathematical accuracy, reasoning structure, and format compliance
        """
        rewards = []
        
        for completion in completions:
            reward = 0.0
            completion_lower = completion.lower()
            
            # 1. Basic structure rewards
            if 10 <= len(completion.split()) <= 150:
                reward += 0.2
            
            # 2. Mathematical vocabulary bonus
            math_terms = [
                'solve', 'calculate', 'equation', 'answer', 'result',
                'step', 'therefore', 'because', 'since', 'so',
                '+', '-', '*', '/', '=', 'equals'
            ]
            math_score = sum(1 for term in math_terms if term in completion_lower)
            reward += min(math_score * 0.05, 0.3)
            
            # 3. Structured reasoning indicators
            structure_indicators = [
                'first', 'second', 'third', 'next', 'then', 'finally',
                'step 1', 'step 2', 'step by step'
            ]
            structure_score = sum(1 for indicator in structure_indicators if indicator in completion_lower)
            reward += min(structure_score * 0.1, 0.25)
            
            # 4. Answer format bonus
            answer_patterns = [
                r'the answer is \d+',
                r'answer: \d+',
                r'= \d+',
                r'therefore.*\d+',
                r'so.*\d+'
            ]
            
            for pattern in answer_patterns:
                if re.search(pattern, completion_lower):
                    reward += 0.3
                    break
            
            # 5. Penalties
            # Too short
            if len(completion.split()) < 5:
                reward -= 0.4
            
            # Too repetitive
            words = completion_lower.split()
            if len(words) > 0:
                unique_ratio = len(set(words)) / len(words)
                if unique_ratio < 0.6:
                    reward -= 0.2
            
            # Nonsensical content
            if any(bad in completion_lower for bad in ['sorry', 'cannot', "don't know", 'unclear']):
                reward -= 0.3
            
            rewards.append(max(reward, -1.0))  # Clamp minimum reward
        
        return rewards
    
    @staticmethod
    def general_quality_reward(completions: List[str], **kwargs) -> List[float]:
        """
        General text quality reward function
        Evaluates coherence, length, and engagement
        """
        rewards = []
        
        for completion in completions:
            reward = 0.0
            
            # Length-based scoring
            word_count = len(completion.split())
            if 20 <= word_count <= 100:
                reward += 0.3
            elif 10 <= word_count < 20:
                reward += 0.1
            elif word_count < 5:
                reward -= 0.4
            elif word_count > 150:
                reward -= 0.2
            
            # Sentence structure
            sentences = [s.strip() for s in completion.split('.') if s.strip()]
            if 2 <= len(sentences) <= 6:
                reward += 0.2
            
            # Vocabulary diversity
            words = completion.lower().split()
            if len(words) > 0:
                unique_ratio = len(set(words)) / len(words)
                reward += min(unique_ratio * 0.3, 0.3)
            
            # Engagement indicators
            engagement_words = [
                'interesting', 'important', 'however', 'although', 'because',
                'therefore', 'furthermore', 'additionally', 'specifically'
            ]
            engagement_score = sum(1 for word in engagement_words if word in completion.lower())
            reward += min(engagement_score * 0.05, 0.2)
            
            rewards.append(reward)
        
        return rewards
    
    @staticmethod
    def create_custom_reward(criteria: Dict[str, Any]) -> Callable:
        """
        Create a custom reward function based on specified criteria
        
        Args:
            criteria: Dictionary with reward criteria
                - min_length: minimum word count
                - max_length: maximum word count  
                - required_terms: list of terms that should appear
                - forbidden_terms: list of terms that should not appear
                - format_bonus: bonus for specific format patterns
        """
        def custom_reward(completions: List[str], **kwargs) -> List[float]:
            rewards = []
            
            for completion in completions:
                reward = 0.0
                words = completion.lower().split()
                word_count = len(words)
                
                # Length constraints
                min_len = criteria.get('min_length', 5)
                max_len = criteria.get('max_length', 200)
                
                if min_len <= word_count <= max_len:
                    reward += 0.3
                elif word_count < min_len:
                    reward -= 0.4
                elif word_count > max_len:
                    reward -= 0.2
                
                # Required terms
                required = criteria.get('required_terms', [])
                for term in required:
                    if term.lower() in completion.lower():
                        reward += 0.2
                
                # Forbidden terms penalty
                forbidden = criteria.get('forbidden_terms', [])
                for term in forbidden:
                    if term.lower() in completion.lower():
                        reward -= 0.3
                
                # Format bonus
                format_patterns = criteria.get('format_bonus', [])
                for pattern in format_patterns:
                    if re.search(pattern, completion.lower()):
                        reward += 0.25
                        break
                
                rewards.append(reward)
            
            return rewards
        
        return custom_reward


class DatasetProcessor:
    """Process and prepare datasets for GRPO training"""
    
    @staticmethod
    def prepare_gsm8k(num_samples: int = 500, split: str = "train") -> Dataset:
        """Prepare GSM8K dataset for math reasoning"""
        from datasets import load_dataset
        
        dataset = load_dataset("gsm8k", "main", split=split)
        
        def format_problem(examples):
            prompts = []
            for question in examples["question"]:
                prompt = f"Solve this math problem step by step:\n\n{question}\n\nSolution:"
                prompts.append(prompt)
            return {"prompt": prompts}
        
        dataset = dataset.map(format_problem, batched=True)
        dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        return dataset
    
    @staticmethod
    def prepare_custom_prompts(prompts: List[str]) -> Dataset:
        """Prepare custom prompts dataset"""
        return Dataset.from_dict({"prompt": prompts})
    
    @staticmethod
    def create_evaluation_set(task_type: str = "math") -> List[str]:
        """Create evaluation prompts for testing"""
        
        if task_type == "math":
            return [
                "What is 15 + 27?",
                "A farmer has 17 sheep. All but 9 die. How many are left?", 
                "If a train travels 60 mph for 2 hours, how far does it go?",
                "Solve for x: 3x + 5 = 20",
                "What is 25% of 80?",
                "A rectangle has length 12 and width 8. What is its area?",
                "If 5 apples cost $3, how much do 8 apples cost?",
                "What is the square root of 144?",
            ]
        else:
            return [
                "Explain the concept of artificial intelligence.",
                "Write a brief description of photosynthesis.",
                "What are the benefits of renewable energy?",
                "Describe how the internet works.",
                "Explain the importance of biodiversity.",
                "What is machine learning?",
                "Describe the water cycle.",
                "Explain supply and demand in economics.",
            ]


class TrainingMonitor:
    """Monitor and visualize training progress"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logs = []
    
    def log_step(self, step: int, metrics: Dict[str, float]):
        """Log training step metrics"""
        log_entry = {"step": step, "timestamp": torch.cuda.Event().record() if torch.cuda.is_available() else None}
        log_entry.update(metrics)
        self.logs.append(log_entry)
    
    def save_logs(self):
        """Save training logs to file"""
        log_file = self.output_dir / "training_logs.json"
        with open(log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)
    
    def plot_training_curves(self):
        """Plot training progress curves"""
        if not self.logs:
            return
        
        steps = [log['step'] for log in self.logs]
        losses = [log.get('loss', 0) for log in self.logs]
        rewards = [log.get('mean_reward', 0) for log in self.logs]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss curve
        ax1.plot(steps, losses, 'b-', linewidth=2)
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True, alpha=0.3)
        
        # Reward curve
        ax2.plot(steps, rewards, 'g-', linewidth=2)
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Mean Reward')
        ax2.set_title('Mean Reward Progress')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()


class ModelEvaluator:
    """Evaluate trained models"""
    
    def __init__(self, tokenizer, base_model, trained_model=None):
        self.tokenizer = tokenizer
        self.base_model = base_model
        self.trained_model = trained_model
        self.device = torch.device("cpu")
    
    def generate_response(self, model, prompt: str, max_tokens: int = 64) -> str:
        """Generate response from a model"""
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(self.tokenizer.decode(inputs[0])):].strip()
    
    def evaluate_on_prompts(self, prompts: List[str]) -> Dict[str, List[str]]:
        """Evaluate both models on a set of prompts"""
        results = {
            "prompts": prompts,
            "base_responses": [],
            "trained_responses": []
        }
        
        for prompt in prompts:
            # Base model response
            base_response = self.generate_response(self.base_model, prompt)
            results["base_responses"].append(base_response)
            
            # Trained model response
            if self.trained_model:
                trained_response = self.generate_response(self.trained_model, prompt)
                results["trained_responses"].append(trained_response)
            else:
                results["trained_responses"].append("No trained model available")
        
        return results
    
    def compute_reward_scores(self, responses: List[str], reward_fn: Callable) -> List[float]:
        """Compute reward scores for responses"""
        return reward_fn(responses)
    
    def compare_models(self, prompts: List[str], reward_fn: Callable) -> Dict[str, Any]:
        """Compare base and trained models"""
        evaluation = self.evaluate_on_prompts(prompts)
        
        base_rewards = self.compute_reward_scores(evaluation["base_responses"], reward_fn)
        trained_rewards = self.compute_reward_scores(evaluation["trained_responses"], reward_fn)
        
        comparison = {
            "evaluation": evaluation,
            "base_rewards": base_rewards,
            "trained_rewards": trained_rewards,
            "mean_base_reward": np.mean(base_rewards),
            "mean_trained_reward": np.mean(trained_rewards),
            "improvement": np.mean(trained_rewards) - np.mean(base_rewards)
        }
        
        return comparison


def setup_cpu_optimization():
    """Setup CPU-specific optimizations"""
    # Disable CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.cuda.is_available = lambda: False
    
    # CPU thread optimization
    torch.set_num_threads(min(8, torch.get_num_threads()))
    
    # Memory optimization
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.enabled = False
    
    print(f"CPU optimization enabled - using {torch.get_num_threads()} threads")


def format_training_summary(logs: List[Dict], model_path: str) -> str:
    """Format training summary for display"""
    if not logs:
        return "No training logs available"
    
    total_steps = len(logs)
    final_loss = logs[-1].get('loss', 0)
    final_reward = logs[-1].get('mean_reward', 0)
    
    summary = f"""
    🎓 Training Summary
    ==================
    
    📊 Total Steps: {total_steps}
    📉 Final Loss: {final_loss:.4f}
    🎯 Final Mean Reward: {final_reward:.4f}
    💾 Model Path: {model_path}
    
    📈 Progress:
    """
    
    # Show progress every 10 steps
    for i, log in enumerate(logs):
        if i % 10 == 0 or i == len(logs) - 1:
            step = log.get('step', i)
            loss = log.get('loss', 0)
            reward = log.get('mean_reward', 0)
            summary += f"    Step {step:3d}: Loss={loss:.4f}, Reward={reward:.4f}\n"
    
    return summary
