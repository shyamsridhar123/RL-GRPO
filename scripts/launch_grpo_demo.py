#!/usr/bin/env python3
"""
Launch script for the GRPO Demo App
Quick launcher for the cleaned-up Gradio interface
"""

import sys
import os
# Add parent directory to import the main app
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import app

if __name__ == "__main__":
    print("🚀 Launching GRPO Demo App...")
    print("📋 Features:")
    print("   ✅ Ultra-fast training via ultra_fast_training.py")
    print("   ✅ GSM8K-style reasoning dataset") 
    print("   ✅ Math reasoning reward function")
    print("   ✅ Model comparison and testing")
    print("   ✅ Performance monitoring")
    print("   ✅ 12-core CPU optimization")
    print("\n🌐 Starting Gradio interface...")
    
    demo = app.create_demo_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )
