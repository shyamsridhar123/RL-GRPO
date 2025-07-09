"""
GRPO Ultra-Fast Training Application - Gradio Interface
CPU-optimized Group Relative Policy Optimization with ultra-fast training
"""

# HARDWARE ACCELERATION SETUP - MUST BE FIRST
import os
import sys

# AGGRESSIVE CPU OPTIMIZATION FOR 12-CORE SYSTEM (matching ultra_fast_training.py)
os.environ['OMP_NUM_THREADS'] = '12'  # Use physical cores
os.environ['MKL_NUM_THREADS'] = '12'
os.environ['NUMEXPR_NUM_THREADS'] = '12'
os.environ['OPENBLAS_NUM_THREADS'] = '12'
os.environ['TORCH_CPP_LOG_LEVEL'] = 'ERROR'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

# Memory optimization - AGGRESSIVE MEMORY CLEANUP
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'  # Smaller chunks
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # Avoid network calls (matching ultra_fast_training.py)
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Aggressive cleanup
os.environ['MALLOC_TRIM_THRESHOLD_'] = '131072'  # Force malloc cleanup

import gradio as gr
import json
import time
from datetime import datetime
from ultra_fast_training import run_ultra_fast_training

class UltraFastGRPODemo:
    """
    Ultra-Fast GRPO demonstration application with Gradio interface
    Uses ultra_fast_training.py exclusively for maximum performance
    """
    
    def __init__(self):
        self.training_logs = []
        self.is_training = False
    
    def start_training(self, dataset_choice: str, task_type: str, num_samples: int, 
                      learning_rate: float, num_epochs: int,
                      progress=gr.Progress()) -> tuple:
        """Start ultra-fast GRPO training"""
        
        if self.is_training:
            return "Training already in progress", json.dumps(self.training_logs, indent=2)
        
        try:
            self.is_training = True
            self.training_logs = []
            
            progress(0.1, desc="🚀 Starting ultra-fast GRPO training...")
            
            # Check system resources first
            import psutil
            memory_info = psutil.virtual_memory()
            available_gb = memory_info.available / 1024**3
            
            if available_gb < 2.0:
                warning_msg = f"⚠️ LOW MEMORY WARNING: Only {available_gb:.1f}GB available. Close other applications for best performance!"
                self.training_logs.append({
                    "message": warning_msg,
                    "timestamp": datetime.now().isoformat()
                })
                print(warning_msg)
            
            self.training_logs.append({
                "message": "🚀 Launching ultra-fast CPU training",
                "timestamp": datetime.now().isoformat()
            })
            
            progress(0.2, desc="⚙️ Configuring training parameters...")
            self.training_logs.append({
                "message": f"📊 Training {num_samples} samples for {num_epochs} epochs",
                "timestamp": datetime.now().isoformat()
            })
            
            self.training_logs.append({
                "message": f"⚙️ Learning rate: {learning_rate}",
                "timestamp": datetime.now().isoformat()
            })
            
            progress(0.3, desc="🏃‍♂️ Running ultra-fast training...")
            
            # Run ultra-fast training
            result = run_ultra_fast_training(
                learning_rate=learning_rate,
                num_samples=int(num_samples),
                num_epochs=int(num_epochs),
                dataset_name=dataset_choice,
                task_type=task_type
            )
            
            progress(0.9, desc="✅ Training completed!")
            
            if result:
                success_message = f"""
                🎊 ULTRA-FAST TRAINING COMPLETED! 🎊
                
                📁 Model Location: {result}
                ⚡ Strategy: ULTRA-FAST CPU OPTIMIZATION
                📊 Samples: {num_samples}
                🔥 Status: ✅ MODEL SAVED SUCCESSFULLY
                """
                
                self.training_logs.append({
                    "message": "✅ Ultra-fast training completed successfully!",
                    "timestamp": datetime.now().isoformat()
                })
                
                self.training_logs.append({
                    "message": f"📁 Model saved at: {result}",
                    "timestamp": datetime.now().isoformat()
                })
                
                progress(1.0, desc="🎉 Training complete!")
                self.is_training = False
                return success_message.strip(), json.dumps(self.training_logs, indent=2)
            else:
                error_msg = "Training failed. Check logs for details."
                self.training_logs.append({
                    "message": f"❌ {error_msg}",
                    "timestamp": datetime.now().isoformat()
                })
                self.is_training = False
                return error_msg, json.dumps(self.training_logs, indent=2)
                
        except Exception as e:
            self.is_training = False
            error_msg = f"Training failed: {str(e)}"
            self.training_logs.append({
                "message": f"❌ {error_msg}",
                "timestamp": datetime.now().isoformat()
            })
            return error_msg, json.dumps(self.training_logs, indent=2)

def create_demo_interface():
    """Create the main Gradio interface"""
    
    demo_app = UltraFastGRPODemo()
    
    # Custom CSS for better styling (matching original app)
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
    
    with gr.Blocks(css=css, title="Ultra-Fast GRPO Training Platform") as demo:
        
        # Main title and description (updated to remove Phase 1 reference)
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1>🧠 Ultra-Fast GRPO Training Demo</h1>
            <p style="font-size: 18px; color: #666;">
                CPU-optimized Group Relative Policy Optimization for Language Models
            </p>
            <p style="color: #888;">
                Train language models using advanced reinforcement learning with maximum speed optimization!
            </p>
            <p style="font-size: 14px; color: #007acc;">
                ⚡ Featuring ultra-fast CPU training with hardware acceleration
            </p>
        </div>
        """)
        
        with gr.Tabs() as tabs:
            
            # Training Configuration Tab
            with gr.Tab("🎯 Ultra-Fast Training", id="training"):
                gr.Markdown("""
                ### Configure Your Ultra-Fast GRPO Training
                
                Set up the training parameters for maximum speed reinforcement learning experiment. 
                This ultra-fast trainer is optimized for CPU and shows improvements quickly.
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
                            minimum=2,
                            maximum=20,
                            value=4,
                            step=2,
                            label="📊 Number of Training Samples",
                            info="Ultra-fast: 2-4 samples for lightning speed, 6-10 for better training"
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
                            info="1 epoch is usually sufficient for ultra-fast training"
                        )
                
                train_button = gr.Button(
                    "⚡ Start Ultra-Fast Training",
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
            
            # Help and Examples Tab
            with gr.Tab("📚 Examples & Help", id="examples"):
                gr.Markdown("""
                ### Ultra-Fast Training Guide
                
                This ultra-fast trainer is optimized for maximum speed CPU training:
                """)
                
                with gr.Accordion("⚡ Ultra-Fast Training Features", open=True):
                    gr.Markdown("""
                    **Ultra-Fast GRPO Training Features:**
                    
                    - ⚡ **Hardware Acceleration**: Optimized for 12-core CPU systems
                    - 🚀 **Lightning Speed**: Minimal dataset for maximum training speed
                    - 💾 **Memory Efficient**: Aggressive memory optimization
                    - 🎯 **CPU Optimized**: Uses all available CPU cores effectively
                    - 📊 **Real-time Monitoring**: Live performance metrics
                    - 🔥 **Fast Convergence**: Shows improvements in seconds, not minutes
                    
                    **Training Tips:**
                    - Start with 4 samples for quick experiments
                    - Use learning rates around 1e-5 for stable training
                    - Math tasks show clear improvement quickly
                    - Monitor logs to see real-time performance metrics
                    """)

                with gr.Accordion("🔢 GSM8K Examples", open=False):
                    gr.Markdown("""
                    **Sample GSM8K Math Problems:**
                    
                    - "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
                    - "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?"
                    - "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents decided to give her twice as much as her parents. How much more money does Betty need to buy the wallet?"
                    
                    The model learns to solve these step-by-step with mathematical reasoning.
                    """)
                
                with gr.Accordion("🛠️ System Requirements", open=False):
                    gr.Markdown("""
                    **Optimized For:**
                    - CPU: 4+ cores (optimized for 12+ cores)
                    - RAM: 8GB minimum, 16GB recommended  
                    - Storage: 5GB free space for models
                    
                    **Performance Tips:**
                    - Close unnecessary applications during training
                    - Use smaller sample sizes for fastest results
                    - Let the system use all CPU cores for maximum speed
                    """)
        
        # Event handlers
        train_button.click(
            demo_app.start_training,
            inputs=[dataset_choice, task_type, num_samples, learning_rate, num_epochs],
            outputs=[training_output, training_logs]
        )
    
    return demo

if __name__ == "__main__":
    # Create and launch the demo
    demo = create_demo_interface()
    
    print("⚡ Starting Ultra-Fast GRPO Training Platform...")
    print("🚀 Optimized for maximum CPU performance...")
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )
