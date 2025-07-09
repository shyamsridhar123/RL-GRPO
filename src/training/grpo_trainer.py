"""
Fixed GRPO Trainer for CPU-only training with comprehensive logging and checkpoint management
Uses correct TRL GRPOConfig parameters and forces CPU usage
"""

import torch
import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments
)
from datasets import Dataset, load_dataset
from trl import GRPOTrainer, GRPOConfig
import numpy as np


@dataclass
class CPUGRPOConfig:
    """Configuration for CPU-optimized GRPO training"""
    # Model settings
    model_name: str = "Qwen/Qwen2-0.5B-Instruct"    # Training settings - using correct TRL parameter names
    output_dir: str = "./grpo_output"
    per_device_train_batch_size: int = 1  # FIXED: Must equal num_generations for GRPO
    gradient_accumulation_steps: int = 2  # Reduced to keep total reasonable
    learning_rate: float = 1e-5
    num_train_epochs: float = 1.0
    warmup_steps: int = 10
    logging_steps: float = 5
    save_steps: float = 25
    eval_steps: Optional[float] = 25
    
    # CPU-specific settings
    use_cpu: bool = True
    no_cuda: bool = True
    bf16: bool = False
    fp16: bool = False
    dataloader_num_workers: int = 1
      # GRPO-specific settings
    max_prompt_length: int = 128
    max_completion_length: int = 64
    num_generations: int = 2  # Must be at least 2 for GRPO
    temperature: float = 0.7
    top_p: float = 0.9
    beta: float = 0.1
    
    # Checkpoint and logging
    save_total_limit: int = 3
    save_safetensors: bool = True
    logging_dir: Optional[str] = None
    report_to: Optional[str] = None  # Set to "tensorboard" if you want TB logging
    
    # Other settings
    seed: int = 42
    disable_dropout: bool = False


class CPULogger:
    """Comprehensive logging system for GRPO training"""
    
    def __init__(self, output_dir: str, level: str = "INFO"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Clear any existing handlers to avoid conflicts
        logger_name = "GRPO_CPU_Training"
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        
        # Setup file and console logging with Windows-friendly encoding
        log_file = self.output_dir / "training.log"
        
        # File handler with UTF-8 encoding
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(getattr(logging, level))
        
        # Console handler with Windows-safe formatting (no emojis)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, level))
        
        # Formatter without emojis for Windows compatibility
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(getattr(logging, level))
        
        self.logger = logger
        
        # Training metrics storage
        self.metrics = []
        self.checkpoints = []
        self.system_info = self._collect_system_info()
        
        # Log initialization without emojis
        self.logger.info("GRPO CPU Training Logger initialized")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Log file: {log_file}")
        self.logger.info(f"System info: {self.system_info}")
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect system information"""
        try:
            import psutil
            cpu_info = {
                "cpu_count": psutil.cpu_count(),
                "cpu_percent": psutil.cpu_percent(),
                "memory_total": psutil.virtual_memory().total / (1024**3),
                "memory_available": psutil.virtual_memory().available / (1024**3)
            }
        except ImportError:
            cpu_info = {
                "cpu_count": os.cpu_count(),
                "note": "Install psutil for detailed system monitoring"
            }
        
        return {
            "python_version": os.sys.version,
            "torch_version": torch.__version__,
            "torch_cuda_available": torch.cuda.is_available(),
            "torch_device": str(torch.device("cpu")),
            **cpu_info        }
    
    def log_training_start(self, config: CPUGRPOConfig, dataset_size: int):
        """Log training start information"""
        self.logger.info("=" * 60)
        self.logger.info("STARTING GRPO TRAINING")
        self.logger.info("=" * 60)
        self.logger.info(f"Model: {config.model_name}")
        self.logger.info(f"Dataset size: {dataset_size}")
        self.logger.info(f"Batch size: {config.per_device_train_batch_size}")
        self.logger.info(f"Learning rate: {config.learning_rate}")
        self.logger.info(f"Epochs: {config.num_train_epochs}")
        self.logger.info(f"Output dir: {config.output_dir}")
        
        # Save config
        config_file = self.output_dir / "training_config.json"
        with open(config_file, 'w') as f:
            json.dump(asdict(config), f, indent=2)
        self.logger.info(f"Configuration saved to {config_file}")
    
    def log_step(self, step: int, logs: Dict[str, float]):
        """Log training step metrics"""
        timestamp = datetime.now().isoformat()
        
        # Create log entry
        log_entry = {
            "step": step,
            "timestamp": timestamp,
            **logs
        }
        self.metrics.append(log_entry)
        
        # Log to console
        loss = logs.get('train_loss', logs.get('loss', 0))
        lr = logs.get('learning_rate', 0)
        
        self.logger.info(f"Step {step:3d}: Loss={loss:.4f}, LR={lr:.2e}")
        
        # Save metrics every 10 steps
        if step % 10 == 0:
            self._save_metrics()
    
    def log_checkpoint(self, step: int, checkpoint_path: str):
        """Log checkpoint creation"""
        checkpoint_info = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "path": checkpoint_path,
            "size_mb": self._get_directory_size(checkpoint_path) if os.path.exists(checkpoint_path) else 0
        }
        self.checkpoints.append(checkpoint_info)
        
        self.logger.info(f"💾 Checkpoint saved at step {step}: {checkpoint_path}")
        self.logger.info(f"📦 Checkpoint size: {checkpoint_info['size_mb']:.1f} MB")
    
    def log_training_end(self, final_model_path: str, training_time: float):
        """Log training completion"""
        self.logger.info("=" * 60)
        self.logger.info("TRAINING COMPLETED")
        self.logger.info("=" * 60)
        self.logger.info(f"⏱️ Total training time: {training_time:.2f} seconds")
        self.logger.info(f"💾 Final model saved to: {final_model_path}")
        
        # Final metrics save
        self._save_metrics()
        self._save_summary(final_model_path, training_time)
    
    def _get_directory_size(self, directory: str) -> float:
        """Get directory size in MB"""
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
            return total_size / (1024 * 1024)  # Convert to MB
        except:
            return 0
    
    def _save_metrics(self):
        """Save training metrics to file"""
        metrics_file = self.output_dir / "training_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def _save_summary(self, final_model_path: str, training_time: float):
        """Save training summary"""
        summary = {
            "training_completed": datetime.now().isoformat(),
            "training_time_seconds": training_time,
            "final_model_path": final_model_path,
            "total_steps": len(self.metrics),
            "checkpoints_created": len(self.checkpoints),
            "final_loss": self.metrics[-1].get('loss', 0) if self.metrics else 0,
            "system_info": self.system_info,
            "checkpoints": self.checkpoints
        }
        
        summary_file = self.output_dir / "training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Training summary saved to {summary_file}")


class CPUGRPOTrainer:
    """
    CPU-optimized GRPO trainer with comprehensive logging and checkpoint management
    """
    def __init__(self, config: CPUGRPOConfig, resume_from: Optional[str] = None):
        self.config = config
        self.device = torch.device("cpu")
        self.resume_from = resume_from
        
        # Force CPU usage at environment level
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["TORCH_USE_CUDA_DSA"] = "0"
        
        # Disable CUDA at PyTorch level
        torch.cuda.is_available = lambda: False
        
        # Setup logging
        self.logger = CPULogger(config.output_dir)
          # Initialize components
        self.tokenizer = None        
        self.model = None
        self.trainer = None
        
        self._setup_model_and_tokenizer()
    
    def _setup_model_and_tokenizer(self):
        """Setup model and tokenizer for CPU training"""
        if self.resume_from:
            self.logger.logger.info(f"Loading model from checkpoint: {self.resume_from}")
            model_path = self.resume_from
        else:
            self.logger.logger.info(f"Loading base model: {self.config.model_name}")
            model_path = self.config.model_name
        
        # Load tokenizer
        try:
            if self.resume_from and os.path.exists(self.resume_from):
                # Try to load tokenizer from checkpoint first
                self.tokenizer = AutoTokenizer.from_pretrained(self.resume_from)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        except:
            # Fallback to base model tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with CPU-specific optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,  # Use float32 for CPU stability
            device_map={"": "cpu"},     # Force CPU mapping
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        # Ensure model is on CPU
        self.model = self.model.cpu()
        
        param_count = sum(p.numel() for p in self.model.parameters())
        self.logger.logger.info(f"Model loaded on device: {next(self.model.parameters()).device}")
        self.logger.logger.info(f"Model parameters: {param_count:,}")
    
    def create_reward_function(self, task_type: str = "math") -> Callable:
        """Create reward function based on task type"""
        
        def math_reward_function(completions: List[str], **kwargs) -> List[float]:
            """Reward function for mathematical reasoning tasks"""
            rewards = []
            
            for completion in completions:
                reward = 0.0
                completion_lower = completion.lower()
                words = completion.split()
                
                # Length-based reward
                if 10 <= len(words) <= 100:
                    reward += 0.3
                elif len(words) < 5:
                    reward -= 0.5
                
                # Mathematical vocabulary
                math_terms = ['answer', 'solution', 'calculate', 'solve', '=', '+', '-', '*', '/', 'step']
                math_count = sum(1 for term in math_terms if term in completion_lower)
                reward += min(math_count * 0.1, 0.4)
                
                # Structured reasoning
                structure_terms = ['first', 'then', 'next', 'therefore', 'so', 'because']
                structure_count = sum(1 for term in structure_terms if term in completion_lower)
                reward += min(structure_count * 0.1, 0.3)
                
                # Answer format bonus
                if any(phrase in completion_lower for phrase in ['the answer is', 'answer:', '= ']):
                    reward += 0.4
                
                # Penalty for poor responses
                if any(bad in completion_lower for bad in ['sorry', 'cannot', "don't know"]):
                    reward -= 0.4
                
                rewards.append(max(reward, -1.0))  # Clamp minimum
            
            return rewards
        
        if task_type == "math":
            return math_reward_function
        else:
            # General reward function
            def general_reward_function(completions: List[str], **kwargs) -> List[float]:
                rewards = []
                for completion in completions:
                    reward = 0.0
                    words = completion.split()
                    
                    # Length reward
                    if 15 <= len(words) <= 80:
                        reward += 0.4
                    elif len(words) < 5:
                        reward -= 0.5
                    
                    # Diversity reward
                    if len(words) > 0:
                        unique_ratio = len(set(words)) / len(words)
                        reward += min(unique_ratio * 0.3, 0.3)
                    
                    rewards.append(reward)
                
                return rewards
            
            return general_reward_function
    
    def prepare_dataset(self, dataset_name: str = "gsm8k", split: str = "train", num_samples: int = 200) -> Dataset:
        """Prepare dataset for GRPO training"""
        self.logger.logger.info(f"📚 Loading {dataset_name} dataset...")
        
        if dataset_name == "gsm8k":
            # Load GSM8K dataset
            dataset = load_dataset("gsm8k", "main", split=split)
            
            def format_gsm8k(examples):
                prompts = []
                for question in examples["question"]:
                    prompt = f"Solve this math problem step by step:\n\n{question}\n\nSolution:"
                    prompts.append(prompt)
                return {"prompt": prompts}
            
            dataset = dataset.map(format_gsm8k, batched=True)
            dataset = dataset.select(range(min(num_samples, len(dataset))))
            
        else:  # Custom prompts
            prompts = [
                "Explain artificial intelligence in simple terms:",
                "Describe the process of photosynthesis:",
                "What are the benefits of renewable energy?",
                "How does machine learning work?",
                "Explain the water cycle:",
            ] * (num_samples // 5 + 1)
            dataset = Dataset.from_dict({"prompt": prompts[:num_samples]})
        
        self.logger.logger.info(f"Dataset prepared with {len(dataset)} samples")
        
        # Validate dataset format
        if len(dataset) > 0:
            sample = dataset[0]
            self.logger.logger.info(f"Dataset sample keys: {list(sample.keys())}")
            self.logger.logger.info(f"Sample prompt: {sample['prompt'][:100]}...")
        
        return dataset
    
    def train(self, dataset: Dataset, reward_fn: Callable, progress_callback: Optional[Callable] = None):
        """Train the model using GRPO with comprehensive logging"""
        
        self.logger.log_training_start(self.config, len(dataset))
        start_time = time.time()
        
        # Setup logging directory
        if self.config.logging_dir is None:
            self.config.logging_dir = str(self.config.output_dir + "/logs")
        
        # Create GRPO training arguments with correct parameters
        training_args = GRPOConfig(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_train_epochs,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            
            # CPU-specific settings
            use_cpu=True,
            no_cuda=True,
            bf16=False,
            fp16=False,
            dataloader_num_workers=self.config.dataloader_num_workers,
            
            # GRPO-specific settings
            max_prompt_length=self.config.max_prompt_length,
            max_completion_length=self.config.max_completion_length,
            num_generations=self.config.num_generations,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            beta=self.config.beta,
            
            # Checkpoint settings
            save_total_limit=self.config.save_total_limit,
            save_safetensors=self.config.save_safetensors,
            logging_dir=self.config.logging_dir,
            report_to=None,  # Disable wandb logging
            
            # Other settings
            seed=self.config.seed,
            disable_dropout=self.config.disable_dropout,
            remove_unused_columns=False,
        )        # Create trainer - FIXED: Pass actual model object, not model name
        self.trainer = GRPOTrainer(
            model=self.model,  # Use the loaded model object
            args=training_args,
            train_dataset=dataset,
            processing_class=self.tokenizer,
            reward_funcs=reward_fn,
        )
        
        # Add custom callback for logging
        class LoggingCallback:
            def __init__(self, logger, progress_callback=None):
                self.logger = logger
                self.progress_callback = progress_callback
                self.step = 0

            def on_train_begin(self, args, state, control, model=None, **kwargs):
                """Called at the beginning of training"""
                self.logger.logger.info("Training started")
            
            def on_epoch_begin(self, args, state, control, model=None, **kwargs):
                """Called at the beginning of each epoch"""
                pass
                
            def on_epoch_end(self, args, state, control, model=None, **kwargs):
                """Called at the end of each epoch"""
                pass
            
            def on_log(self, args, state, control, model=None, logs=None, **kwargs):
                if logs:
                    self.step = state.global_step
                    self.logger.log_step(self.step, logs)
                    
                    if self.progress_callback:
                        self.progress_callback({
                            'step': self.step,
                            'loss': logs.get('train_loss', logs.get('loss', 0)),
                            'learning_rate': logs.get('learning_rate', 0),
                            'timestamp': datetime.now().isoformat()
                        })
            
            def on_save(self, args, state, control, model=None, **kwargs):
                checkpoint_path = f"{args.output_dir}/checkpoint-{state.global_step}"
                self.logger.log_checkpoint(state.global_step, checkpoint_path)        # Train the model
        try:
            self.logger.logger.info("Starting GRPO training...")
            self.logger.logger.info(f"Dataset length: {len(dataset)}")
            self.logger.logger.info(f"Expected steps: {len(dataset) // (self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps)}")
            
            # Add debugging - check trainer state
            self.logger.logger.info(f"Trainer created successfully")
            self.logger.logger.info(f"Training arguments: {training_args}")
            
            # Train the model
            training_result = self.trainer.train()
            self.logger.logger.info(f"Training completed. Result: {training_result}")
            
            # Save final model
            final_model_path = os.path.join(self.config.output_dir, "final_model")
            self.trainer.save_model(final_model_path)
            
            training_time = time.time() - start_time
            self.logger.log_training_end(final_model_path, training_time)
            
            return final_model_path
            
        except Exception as e:
            self.logger.logger.error(f"Training failed with error: {str(e)}")
            self.logger.logger.error(f"Error type: {type(e).__name__}")
            import traceback
            self.logger.logger.error(f"Full traceback: {traceback.format_exc()}")
            raise e
    
    def load_trained_model(self, model_path: str):
        """Load a trained model for inference"""
        self.logger.logger.info(f"📥 Loading trained model from: {model_path}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map={"": "cpu"},
            low_cpu_mem_usage=True
        )
        
        self.logger.logger.info("Trained model loaded successfully")
    
    def generate_response(self, prompt: str, max_new_tokens: int = 64, temperature: float = 0.7) -> str:
        """Generate response using the current model"""
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the input prompt from response
        response = response[len(prompt):].strip()
        
        return response
