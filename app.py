"""
GRPO Demo Application - Ultra-Fast Training Interface
CPU-optimized Group Relative Policy Optimization using ultra_fast_training.py
"""

# HARDWARE ACCELERATION SETUP - MUST BE FIRST
import os
import sys

# AGGRESSIVE CPU OPTIMIZATION FOR 12-CORE SYSTEM
os.environ['OMP_NUM_THREADS'] = '12'  # Use physical cores
os.environ['MKL_NUM_THREADS'] = '12'
os.environ['NUMEXPR_NUM_THREADS'] = '12'
os.environ['OPENBLAS_NUM_THREADS'] = '12'
os.environ['TORCH_CPP_LOG_LEVEL'] = 'ERROR'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

# Memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['TRANSFORMERS_OFFLINE'] = '0'  # Allow network for model downloads

import gradio as gr
import torch
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Configure PyTorch for CPU optimization
torch.set_num_threads(12)
torch.manual_seed(42)

from transformers import AutoTokenizer, AutoModelForCausalLM

# Import ultra_fast_training functions directly for maximum speed
try:
    from ultra_fast_training import run_ultra_fast_training
    DIRECT_TRAINING_AVAILABLE = True
    print("✅ Direct ultra-fast training available")
except ImportError as e:
    print(f"⚠️ Direct training import failed: {e}")
    DIRECT_TRAINING_AVAILABLE = False


class GRPODemo:
    """
    Main GRPO demonstration application with Gradio interface
    Uses ultra_fast_training.py for actual training
    """
    
    def __init__(self):
        # Model components
        self.base_model = None
        self.tokenizer = None
        self.training_logs = []
        self.is_training = False
        
        # Base model configuration
        self.base_model_name = "Qwen/Qwen2-0.5B-Instruct"
        
        # Initialize base model for comparison
        self._load_base_model()
        
        # Available checkpoints and models
        self.available_models = self._scan_available_models()
    
    def _load_base_model(self):
        """Load base model for before/after comparison"""
        try:
            print("Loading base model for comparison...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True
            )
            
            print("Base model loaded successfully")
        except Exception as e:
            print(f"Error loading base model: {e}")
    
    def generate_base_response(self, prompt: str, max_tokens: int = 64, temperature: float = 0.7) -> str:
        """Generate response using base model"""
        if self.base_model is None:
            return "Base model not loaded"
        
        try:
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.base_model.generate(
                    inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response[len(prompt):].strip()
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def generate_trained_response(self, prompt: str, max_tokens: int = 64, temperature: float = 0.7) -> str:
        """Generate response using trained model"""
        return "No trained model interface available. Please use model comparison feature."
    
    def start_training(self, dataset_choice: str, task_type: str, num_samples: int, 
                      learning_rate: float, num_epochs: int, checkpoint_choice: str = "Base Model (Fresh Start)",
                      progress=gr.Progress()) -> tuple:
        """Start GRPO training using ultra_fast_training.py"""
        
        if self.is_training:
            return "Training already in progress", json.dumps(self.training_logs, indent=2)
        
        try:
            self.is_training = True
            self.training_logs = []
            
            progress(0.1, desc="🚀 Starting ultra-fast GRPO training...")
            self.training_logs.append({
                "message": "🚀 Using ultra_fast_training.py for actual training",
                "timestamp": datetime.now().isoformat()
            })
            
            # Use the working ultra_fast_training.py script
            return self._run_ultra_fast_training(
                dataset_choice, task_type, num_samples, 
                learning_rate, num_epochs, checkpoint_choice, progress
            )
                
        except Exception as e:
            self.is_training = False
            error_msg = f"Training failed: {str(e)}"
            self.training_logs.append({
                "message": f"❌ {error_msg}",
                "timestamp": datetime.now().isoformat()
            })
            return error_msg, json.dumps(self.training_logs, indent=2)
    
    def _run_ultra_fast_training(self, dataset_choice: str, task_type: str, num_samples: int,
                                learning_rate: float, num_epochs: int, checkpoint_choice: str,
                                progress=gr.Progress()) -> tuple:
        """Run training using direct function calls for maximum speed"""
        
        progress(0.1, desc="⚡ Starting DIRECT ultra-fast training...")
        
        self.training_logs.append({
            "message": f"⚡ Using DIRECT ultra_fast_training function (NO SUBPROCESS OVERHEAD)",
            "timestamp": datetime.now().isoformat()
        })
        
        self.training_logs.append({
            "message": f"📊 Training {num_samples} samples for {num_epochs} epochs",
            "timestamp": datetime.now().isoformat()
        })
        
        self.training_logs.append({
            "message": f"⚙️ Learning rate: {learning_rate}",
            "timestamp": datetime.now().isoformat()
        })
        
        progress(0.2, desc="🚀 Running ultra-fast training directly...")
        
        # Use direct function call instead of subprocess for maximum speed
        if DIRECT_TRAINING_AVAILABLE:
            return self._run_direct_training(
                dataset_choice, task_type, num_samples, 
                learning_rate, num_epochs, checkpoint_choice, progress
            )
        else:
            # Fallback to subprocess if direct import failed
            return self._run_subprocess_training(
                dataset_choice, task_type, num_samples, 
                learning_rate, num_epochs, checkpoint_choice, progress
            )
    
    def _run_direct_training(self, dataset_choice: str, task_type: str, num_samples: int,
                           learning_rate: float, num_epochs: int, checkpoint_choice: str,
                           progress=gr.Progress()) -> tuple:
        """Run training using direct function calls - MAXIMUM SPEED"""
        
        progress(0.3, desc="⚡ Starting DIRECT training...")
        
        try:
            # Log start
            start_time = time.time()
            self.training_logs.append({
                "message": "⚡ Starting DIRECT ultra-fast training (NO SUBPROCESS)",
                "timestamp": datetime.now().isoformat()
            })
            
            progress(0.4, desc="🔥 Running ultra-fast training...")
            
            # Call the training function DIRECTLY - no subprocess overhead!
            final_model_path = run_ultra_fast_training(
                learning_rate=learning_rate,
                num_samples=num_samples,
                num_epochs=num_epochs,
                dataset_name=dataset_choice,
                task_type=task_type
            )
            
            training_time = time.time() - start_time
            
            progress(0.9, desc="✅ Direct training completed!")
            
            self.training_logs.append({
                "message": f"✅ DIRECT training completed in {training_time:.2f} seconds!",
                "timestamp": datetime.now().isoformat()
            })
            
            if final_model_path:
                self.training_logs.append({
                    "message": f"💾 Model saved to: {final_model_path}",
                    "timestamp": datetime.now().isoformat()
                })
                
                # Refresh available models
                self.available_models = self._scan_available_models()
                
                progress(1.0, desc="🎉 Training completed successfully!")
                
                success_msg = f"🎉 DIRECT ultra-fast training completed in {training_time:.2f}s!\n\n✅ Model saved to: {final_model_path}\n\n⚡ DIRECT FUNCTION CALL - MAXIMUM SPEED!"
                return success_msg, json.dumps(self.training_logs, indent=2)
            else:
                error_msg = "❌ Training completed but no model path returned"
                self.training_logs.append({
                    "message": error_msg,
                    "timestamp": datetime.now().isoformat()
                })
                return error_msg, json.dumps(self.training_logs, indent=2)
                
        except Exception as e:
            error_msg = f"❌ Direct training failed: {str(e)}"
            self.training_logs.append({
                "message": error_msg,
                "timestamp": datetime.now().isoformat()
            })
            
            # Add detailed error info
            import traceback
            self.training_logs.append({
                "message": f"🔍 Error details: {traceback.format_exc()}",
                "timestamp": datetime.now().isoformat()
            })
            
            return error_msg, json.dumps(self.training_logs, indent=2)
        
        finally:
            self.is_training = False
    
    def _run_subprocess_training(self, dataset_choice: str, task_type: str, num_samples: int,
                               learning_rate: float, num_epochs: int, checkpoint_choice: str,
                               progress=gr.Progress()) -> tuple:
        """Fallback: Run training using subprocess (SLOWER but compatible)"""
        
        progress(0.3, desc="� Starting subprocess training (slower)...")
        
        self.training_logs.append({
            "message": "🐌 Using subprocess training (slower - direct import failed)",
            "timestamp": datetime.now().isoformat()
        })
        
        # Build command for ultra_fast_training.py
        cmd = [
            sys.executable, "ultra_fast_training.py",
            "--num_samples", str(num_samples),
            "--learning_rate", str(learning_rate),
            "--num_epochs", str(num_epochs),
            "--dataset", dataset_choice,
            "--task_type", task_type
        ]
        
        # Add checkpoint if not starting fresh
        if checkpoint_choice != "Base Model (Fresh Start)":
            if checkpoint_choice in self.available_models:
                checkpoint_path = self.available_models[checkpoint_choice]
                if checkpoint_path:  # Only add if path is not None
                    cmd.extend(["--checkpoint", checkpoint_path])
        
        return self._execute_training_process(cmd, "ultra_fast", progress)
    
    def _execute_training_process(self, cmd: list, strategy: str, progress=gr.Progress()) -> tuple:
        """Execute the training process using ultra_fast_training.py"""
        
        progress(0.6, desc=f"🚀 Starting {strategy} training process...")
        
        self.training_logs.append({
            "message": f"� Executing: {' '.join(cmd[:3])} ...",
            "timestamp": datetime.now().isoformat()
        })
        
        # Start the process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            universal_newlines=True
        )
        
        # Monitor the training progress
        output_lines = []
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                line = output.strip()
                output_lines.append(line)
                
                # Update progress based on output
                if "Training completed" in line:
                    progress(0.9, desc="Training completed!")
                elif "Step" in line:
                    progress(0.7, desc="Training in progress...")
                elif "Saving" in line:
                    progress(0.8, desc="Saving model...")
                
                # Log important messages
                if any(keyword in line.lower() for keyword in ["step", "loss", "training", "completed", "saved"]):
                    self.training_logs.append({
                        "message": line,
                        "timestamp": datetime.now().isoformat()
                    })
        
        return_code = process.poll()
        self.is_training = False
        
        if return_code == 0:
            progress(1.0, desc="✅ Training completed successfully!")
            self.training_logs.append({
                "message": f"✅ Ultra-fast training completed successfully!",
                "timestamp": datetime.now().isoformat()
            })
            
            # Refresh available models
            self.available_models = self._scan_available_models()
            
            success_msg = f"✅ GRPO training completed successfully using ultra_fast_training.py!"
            return success_msg, json.dumps(self.training_logs, indent=2)
        else:
            error_msg = f"❌ Training failed with return code {return_code}"
            self.training_logs.append({
                "message": error_msg,
                "timestamp": datetime.now().isoformat()
            })
            return error_msg, json.dumps(self.training_logs + [{"message": line} for line in output_lines], indent=2)
    def get_training_status(self) -> str:
        """Get current training status"""
        if self.is_training:
            return "🔄 Training in progress...\n\nPlease wait while the model is being trained. This may take several minutes on CPU."
        else:
            return "✅ Ready for training"
    
    def compare_models(self, prompt: str, base_model_choice: str, trained_model_choice: str, max_tokens: int = 64, temperature: float = 0.7) -> tuple:
        """Compare selected base model vs trained model responses"""
        
        base_response = self.generate_selected_response(prompt, base_model_choice, max_tokens, temperature)
        trained_response = self.generate_selected_response(prompt, trained_model_choice, max_tokens, temperature)
        return base_response, trained_response
    
    def generate_selected_response(self, prompt: str, model_choice: str, max_tokens: int = 64, temperature: float = 0.7) -> str:
        """Generate response using selected model"""
        try:
            # Handle original base model
            if model_choice == "Qwen/Qwen2-0.5B-Instruct (Original Base)":
                return self.generate_base_response(prompt, max_tokens, temperature)
            
            # Handle other models by loading them temporarily
            if model_choice in self.available_models:
                model_path = self.available_models[model_choice]
                if model_path is None:
                    return self.generate_base_response(prompt, max_tokens, temperature)
                
                # Load the model temporarily for inference
                try:
                    temp_tokenizer = AutoTokenizer.from_pretrained(model_path)
                    if temp_tokenizer.pad_token is None:
                        temp_tokenizer.pad_token = temp_tokenizer.eos_token
                    
                    temp_model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float32,
                        device_map="cpu",
                        low_cpu_mem_usage=True
                    )
                    
                    # Generate response
                    inputs = temp_tokenizer.encode(prompt, return_tensors="pt")
                    with torch.no_grad():
                        outputs = temp_model.generate(
                            inputs,
                            max_new_tokens=max_tokens,
                            temperature=temperature,
                            top_p=0.9,
                            do_sample=True,
                            pad_token_id=temp_tokenizer.eos_token_id,
                        )
                    
                    response = temp_tokenizer.decode(outputs[0], skip_special_tokens=True)
                    return response[len(prompt):].strip()
                    
                except Exception as e:
                    return f"Error loading model {model_choice}: {str(e)}"
            else:
                return f"Model '{model_choice}' not found in available models."
                
        except Exception as e:
            return f"Error generating response with {model_choice}: {str(e)}"
    
    def get_training_status(self) -> str:
        """Get current training status"""
        if self.is_training:
            return "🔄 Training in progress...\n\nPlease wait while the model is being trained. This may take several minutes on CPU."
        else:
            # Check if models were saved successfully
            output_dirs = [
                "./models/grpo_output/final_model", 
                "./models/grpo_extended/final_model",
                "./models/ultra_fast/final_model",
                "./models/hardware_accelerated/final_model",
                "./models/extreme_fast/final_model",
                "./models/stage1/final_model",
                "./models/stage2/final_model",
                "./models/stage3/final_model"
            ]
            saved_models = []
            for dir_path in output_dirs:
                if os.path.exists(dir_path):
                    saved_models.append(dir_path)
            
            if saved_models:
                status = f"✅ Training system ready!\n\n"
                status += f"Found {len(saved_models)} trained models available:\n"
                for model in saved_models[:3]:  # Show first 3
                    status += f"• {model}\n"
                if len(saved_models) > 3:
                    status += f"... and {len(saved_models)-3} more\n"
                status += "\nYou can start new training or test existing models."
                return status
            else:
                return "⏳ Ready to start training\n\nClick 'Start GRPO Training' to begin training a model with ultra_fast_training.py."
    
    def get_recent_training_summary(self) -> str:
        """Get a summary of recent training activity"""
        if not self.training_logs:
            return "No training activity recorded yet."
        
        summary = f"📊 Training Summary ({len(self.training_logs)} entries)\n\n"
        
        if self.training_logs:
            # Show last few entries
            recent_logs = self.training_logs[-5:] if len(self.training_logs) > 5 else self.training_logs
            
            for log in recent_logs:
                if isinstance(log, dict):
                    step = log.get('step', 'N/A')
                    loss = log.get('loss', 0)
                    summary += f"• Step {step}: Loss = {loss:.4f}\n"
                else:
                    summary += f"• {str(log)[:100]}...\n"
            
            if len(self.training_logs) > 5:
                summary += f"\n... and {len(self.training_logs) - 5} more entries"
        
        return summary
    
    def get_training_logs(self) -> str:
        """Get current training logs"""
        if self.training_logs:
            # Format logs for better readability
            formatted_logs = []
            for i, log in enumerate(self.training_logs):
                if isinstance(log, dict):
                    formatted_logs.append({
                        'entry': i + 1,
                        'step': log.get('step', 'N/A'),
                        'loss': f"{log.get('loss', 0):.4f}" if log.get('loss') else 'N/A',
                        'learning_rate': f"{log.get('learning_rate', 0):.2e}" if log.get('learning_rate') else 'N/A',
                        'timestamp': log.get('timestamp', 'N/A'),
                        'message': log.get('message', str(log))
                    })
                else:
                    formatted_logs.append({
                        'entry': i + 1,
                        'raw_data': str(log)
                    })
            
            return json.dumps(formatted_logs, indent=2)
        else:
            return "No training logs available"
    
    def _scan_available_models(self) -> Dict[str, str]:
        """Scan for available final trained models only (no checkpoints)"""
        models = {"Base Model (Fresh Start)": None}
        
        # Progressive training stages (priority order) - only final models
        progressive_stages = [
            ("📊 Stage 1: Basic Math", "./models/stage1/final_model"),
            ("📈 Stage 2: Intermediate Math", "./models/stage2/final_model"),
            ("🎯 Stage 3: Advanced Math", "./models/stage3/final_model"),
            ("⚡ Ultra-Fast Model", "./models/ultra_fast/final_model"),
            ("🔥 Hardware Accelerated", "./models/hardware_accelerated/final_model"),
            ("⚡ Extreme Fast Model", "./models/extreme_fast/final_model")
        ]
        
        for stage_name, stage_path in progressive_stages:
            if os.path.exists(stage_path):
                # Check if it contains model files
                has_model_files = any(
                    os.path.exists(os.path.join(stage_path, f)) 
                    for f in ["pytorch_model.bin", "model.safetensors", "config.json"]
                )
                if has_model_files:
                    models[stage_name] = stage_path
        
        # Other common final model directories - only include final_model subdirectories
        other_model_dirs = [
            "./grpo_output/final_model",
            "./grpo_extended/final_model"
        ]
        
        for model_dir in other_model_dirs:
            if os.path.exists(model_dir):
                # Check if it contains model files
                has_model_files = any(
                    os.path.exists(os.path.join(model_dir, f)) 
                    for f in ["pytorch_model.bin", "model.safetensors", "config.json"]
                )
                
                if has_model_files:
                    # Create a friendly name
                    parent_name = os.path.basename(os.path.dirname(model_dir))
                    friendly_name = f"{parent_name}/final_model"
                    models[f"📁 {friendly_name}"] = model_dir
        
        # Scan for any other final_model directories in the models folder
        models_base_path = "./models"
        if os.path.exists(models_base_path):
            for item in os.listdir(models_base_path):
                item_path = os.path.join(models_base_path, item)
                if os.path.isdir(item_path):
                    final_model_path = os.path.join(item_path, "final_model")
                    if os.path.exists(final_model_path):
                        # Check if it contains model files
                        has_model_files = any(
                            os.path.exists(os.path.join(final_model_path, f)) 
                            for f in ["pytorch_model.bin", "model.safetensors", "config.json"]
                        )
                        if has_model_files:
                            # Skip if already added in progressive stages
                            stage_key = f"📁 {item}/final_model"
                            if stage_key not in models:
                                models[stage_key] = final_model_path
        
        return models
    
    def refresh_available_models(self) -> tuple:
        """Refresh and return updated choices for all model dropdowns"""
        self.available_models = self._scan_available_models()
        choices = list(self.available_models.keys())
        
        # For base model dropdown (includes original base)
        base_choices = ["Qwen/Qwen2-0.5B-Instruct (Original Base)"] + [k for k in choices if k != "Base Model (Fresh Start)"]
        
        # For trained model dropdown (excludes original base and fresh start)
        trained_choices = [k for k in choices if k != "Base Model (Fresh Start)"]
        
        return (
            gr.update(choices=base_choices),
            gr.update(choices=trained_choices),
            gr.update(choices=choices)
        )
    
    def load_checkpoint_for_training(self, model_choice: str) -> str:
        """Load a specific model/checkpoint for continued training"""
        if model_choice == "Base Model (Fresh Start)" or model_choice not in self.available_models:
            return "Using fresh base model for training"
        
        model_path = self.available_models[model_choice]
        if model_path and os.path.exists(model_path):
            return f"✅ Checkpoint available: {model_path}"
        else:
            return f"❌ Checkpoint not found: {model_choice}"
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            print("✅ Cleanup completed")
        except Exception as e:
            print(f"⚠️  Error during cleanup: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()
    
def create_demo_interface():
    """Create the main Gradio interface"""
    
    demo_app = GRPODemo()
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .training-status {
        font-weight: bold;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .success {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .error {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    """
    
    with gr.Blocks(css=css, title="GRPO CPU Demo Platform") as demo:
        
        # Main title and description
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1>🧠 Ultra-Fast GRPO Training Platform</h1>
            <p style="font-size: 18px; color: #666;">
                ⚡ MAXIMUM SPEED: Direct Function Calls + 12-Core CPU Optimization
            </p>
            <p style="color: #888;">
                Train and compare language models using advanced reinforcement learning - no GPU required!
            </p>
            <p style="font-size: 14px; color: #007acc;">
                🚀 NEW: Eliminates subprocess overhead for ultra-fast training
            </p>
        </div>
        """)
        
        # Training Status Display (NO PHASE 1 GARBAGE)
        with gr.Row():
            with gr.Column():
                status_display = gr.Textbox(
                    value=demo_app.get_training_status(),
                    label="📊 Training Status",
                    interactive=False,
                    lines=6
                )
        
        with gr.Tabs() as tabs:
            
            # Training Configuration Tab
            with gr.Tab("🎯 Training Setup", id="training"):
                gr.Markdown("""
                ### Configure Your GRPO Training
                
                Set up the training parameters for your reinforcement learning experiment using ultra_fast_training.py.
                GRPO is optimized for CPU training and typically shows improvements within 30 training steps.
                """)
                
                with gr.Row():
                    with gr.Column():
                        dataset_choice = gr.Dropdown(
                            choices=["gsm8k", "custom"],
                            value="gsm8k",
                            label="📚 Dataset",
                            info="GSM8K for math problems, Custom for general text"
                        )
                        
                        task_type = gr.Dropdown(
                            choices=["math", "general"],
                            value="math",
                            label="🎲 Task Type",
                            info="Determines the reward function used"
                        )
                        
                        num_samples = gr.Slider(
                            minimum=5,
                            maximum=100,
                            value=10,
                            step=5,
                            label="📊 Number of Training Samples",
                            info="Small datasets for quick CPU training (5-50 recommended)"
                        )                    
                    with gr.Column():
                        learning_rate = gr.Number(
                            value=1e-5,
                            label="📈 Learning Rate",
                            info="Lower values are more stable for CPU training"
                        )
                        
                        num_epochs = gr.Slider(
                            minimum=1,
                            maximum=3,
                            value=1,
                            step=1,
                            label="🔄 Training Epochs",
                            info="1 epoch is usually sufficient for GRPO"
                        )
                        
                        checkpoint_choice = gr.Dropdown(
                            choices=list(demo_app.available_models.keys()),
                            value="Base Model (Fresh Start)",
                            label="🎯 Starting Point",
                            info="Continue from existing model or start fresh"
                        )
                
                with gr.Row():
                    refresh_models_btn = gr.Button("🔄 Refresh Available Models", size="sm")
                    checkpoint_status = gr.Textbox(
                        label="📋 Checkpoint Status",
                        interactive=False,
                        lines=2
                    )
                
                train_button = gr.Button(
                    "🚀 Start GRPO Training",
                    variant="primary",
                    size="lg"
                )
                
                training_output = gr.Textbox(
                    label="📋 Training Status",
                    lines=8,
                    placeholder="Training status will appear here..."
                )
                
                with gr.Accordion("📈 Training Logs", open=False):
                    training_logs = gr.Code(
                        label="Detailed Training Logs",
                        language="json",
                        lines=15
                    )
            
            # Model Testing Tab
            with gr.Tab("🧪 Model Testing", id="testing"):
                gr.Markdown("""
                ### Test and Compare Models
                
                Choose any base model and trained model to compare their performance.
                Select from available progressive training stages or any saved checkpoints.
                """)
                
                with gr.Row():
                    with gr.Column():
                        # Model Selection Section
                        gr.Markdown("#### 🎯 Model Selection")
                        
                        base_model_choice = gr.Dropdown(
                            choices=["Qwen/Qwen2-0.5B-Instruct (Original Base)"] + [k for k in demo_app.available_models.keys() if k != "Base Model (Fresh Start)"],
                            value="Qwen/Qwen2-0.5B-Instruct (Original Base)",
                            label="� Base Model",
                            info="Choose the baseline model for comparison"
                        )
                        
                        trained_model_choice = gr.Dropdown(
                            choices=[k for k in demo_app.available_models.keys() if k != "Base Model (Fresh Start)"],
                            value=list(demo_app.available_models.keys())[1] if len(demo_app.available_models) > 1 else None,
                            label="✨ Trained Model",
                            info="Choose the trained model to test"
                        )
                        
                        refresh_models_btn2 = gr.Button("🔄 Refresh Available Models", size="sm")
                        
                    with gr.Column():
                        # Generation Parameters
                        gr.Markdown("#### ⚙️ Generation Settings")
                        
                        with gr.Row():
                            max_tokens = gr.Slider(
                                minimum=10,
                                maximum=200,
                                value=64,
                                label="📏 Max New Tokens"
                            )
                            
                            temperature = gr.Slider(
                                minimum=0.1,
                                maximum=2.0,
                                value=0.7,
                                label="🌡️ Temperature"
                            )
                
                # Prompt Input
                prompt_input = gr.Textbox(
                    label="💬 Enter Your Prompt",
                    placeholder="Example: Solve this math problem: What is 15 + 27?",
                    lines=3
                )
                
                with gr.Row():
                    test_base_button = gr.Button("🎯 Test Base Model Only", variant="secondary")
                    test_trained_button = gr.Button("✨ Test Trained Model Only", variant="secondary")
                    compare_button = gr.Button("⚖️ Compare Both Models", variant="primary")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 📝 Base Model Response")
                        base_response = gr.Textbox(
                            label="Before Training",
                            lines=6,
                            placeholder="Base model response will appear here..."
                        )
                    
                    with gr.Column():
                        gr.Markdown("#### ✨ Trained Model Response")
                        trained_response = gr.Textbox(
                            label="After GRPO Training",
                            lines=6,
                            placeholder="Trained model response will appear here..."
                        )
            
            # Examples and Help Tab
            with gr.Tab("📚 Examples & Help", id="examples"):
                gr.Markdown("""
                ### Example Prompts and Usage Guide
                
                Try these example prompts to see the difference between base and trained models:
                """)
                
                gr.Examples(
                    examples=[
                        ["Solve this math problem step by step: What is 25 + 37?"],
                        ["A farmer has 20 sheep. All but 9 die. How many sheep are left?"],
                        ["If a train travels at 80 mph for 1.5 hours, how far does it travel?"],
                        ["Solve for x: 2x + 10 = 30"],
                        ["What is 15% of 200?"],
                        ["A rectangle has length 8 and width 5. What is its area?"]
                    ],
                    inputs=[prompt_input],
                    label="🧮 Math Problem Examples"
                )
                
                with gr.Accordion("ℹ️ How GRPO Works", open=False):
                    gr.Markdown("""
                    **Group Relative Policy Optimization (GRPO)** is an advanced reinforcement learning algorithm that:
                    
                    - ✅ **No Value Function**: Unlike PPO, GRPO doesn't need a separate value function
                    - ✅ **Relative Rewards**: Uses rewards relative to batch averages for better stability  
                    - ✅ **CPU Optimized**: Designed to work efficiently on CPU hardware
                    - ✅ **Memory Efficient**: Lower memory footprint than traditional RL methods
                    - ✅ **Fast Convergence**: Often shows improvements within 20-30 training steps
                    
                    **Training Tips:**
                    - Start with small datasets (5-20 samples) for quick experiments
                    - Use learning rates around 1e-5 for stable CPU training
                    - Math tasks typically show clear improvement with GRPO
                    - Monitor the training logs to see the "aha moment" when performance jumps
                    - ultra_fast_training.py is optimized for 12-core CPU systems
                    """)
                
                with gr.Accordion("🛠️ Troubleshooting", open=False):
                    gr.Markdown("""
                    **Common Issues and Solutions:**
                    
                    - **Training takes too long**: Reduce number of samples to 5-20
                    - **Out of memory**: Training auto-adjusts batch size based on available memory
                    - **Poor model performance**: Try increasing samples gradually (10 → 20 → 50)
                    - **Training fails**: Check ultra_fast_training.py output for detailed error messages
                    
                    **System Requirements:**
                    - CPU: 4+ cores (optimized for 12+ cores)
                    - RAM: 4GB minimum, 8GB recommended  
                    - Storage: 3GB free space for models and outputs
                    - Python packages: transformers, torch, datasets, psutil
                    """)
          # Add refresh buttons for status and logs
        with gr.Row():
            refresh_status_btn = gr.Button("🔄 Refresh Status", size="sm")
            refresh_logs_btn = gr.Button("📋 Refresh Logs", size="sm")
        # Event handlers
        train_button.click(
            demo_app.start_training,
            inputs=[dataset_choice, task_type, num_samples, learning_rate, num_epochs, checkpoint_choice],
            outputs=[training_output, training_logs]
        ).then(
            demo_app.get_training_status,
            outputs=[status_display]
        )
        
        compare_button.click(
            demo_app.compare_models,
            inputs=[prompt_input, base_model_choice, trained_model_choice, max_tokens, temperature],
            outputs=[base_response, trained_response]
        )
        
        test_base_button.click(
            demo_app.generate_selected_response,
            inputs=[prompt_input, base_model_choice, max_tokens, temperature],
            outputs=[base_response]
        )
        
        test_trained_button.click(
            demo_app.generate_selected_response,
            inputs=[prompt_input, trained_model_choice, max_tokens, temperature],
            outputs=[trained_response]
        )
          # Refresh functionality
        refresh_status_btn.click(
            demo_app.get_training_status,
            outputs=[status_display]
        )
        
        refresh_logs_btn.click(
            demo_app.get_training_logs,
            outputs=[training_logs]
        )
        
        # Refresh models functionality
        refresh_models_btn.click(
            demo_app.refresh_available_models,
            outputs=[base_model_choice, trained_model_choice, checkpoint_choice]
        )
        
        refresh_models_btn2.click(
            demo_app.refresh_available_models,
            outputs=[base_model_choice, trained_model_choice, checkpoint_choice]
        )
        
        # Update checkpoint status when selection changes
        checkpoint_choice.change(
            demo_app.load_checkpoint_for_training,
            inputs=[checkpoint_choice],
            outputs=[checkpoint_status]
        )
    
    return demo


if __name__ == "__main__":
    # Create and launch the demo
    demo = create_demo_interface()
    
    print("🚀 Starting GRPO CPU Demo Platform...")
    print("📝 This may take a moment to load the base model...")
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )
