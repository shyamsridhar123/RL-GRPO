#!/usr/bin/env python3
"""
GRPO Training Results Visualizer
Generates plots and visualizations from training logs and metrics
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TrainingVisualizer:
    """Visualize GRPO training results from logs and metrics"""
    
    def __init__(self, log_dir: str, output_dir: str = None):
        """
        Initialize the training visualizer
        
        Args:
            log_dir: Directory containing training logs
            output_dir: Directory to save visualizations (defaults to log_dir/viz)
        """
        self.log_dir = Path(log_dir)
        self.output_dir = Path(output_dir) if output_dir else self.log_dir / "viz"
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_training_logs(self, filename: str) -> Dict:
        """Load training logs from file"""
        log_path = self.log_dir / filename
        
        if not log_path.exists():
            raise FileNotFoundError(f"Log file not found: {log_path}")
        
        with open(log_path, 'r') as f:
            return json.load(f)
    
    def load_monitoring_logs(self) -> List[Dict]:
        """Load all monitoring logs"""
        monitoring_logs = []
        
        for file_path in self.log_dir.glob("monitoring_*.json"):
            try:
                with open(file_path, 'r') as f:
                    monitoring_logs.append(json.load(f))
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        return monitoring_logs
    
    def visualize_training_loss(self, log_data: Dict, save_path: Optional[str] = None):
        """Visualize training loss over time"""
        if "metrics" not in log_data:
            print("No metrics found in log data")
            return
        
        # Extract loss values and steps
        steps = []
        losses = []
        
        for entry in log_data["metrics"]:
            if "step" in entry and ("loss" in entry or "train_loss" in entry):
                steps.append(entry.get("step", 0))
                losses.append(entry.get("loss", entry.get("train_loss", 0)))
        
        if not steps:
            print("No loss data found in logs")
            return
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(steps, losses, 'b-', linewidth=2)
        plt.title("GRPO Training Loss", fontsize=16)
        plt.xlabel("Step", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Save or display
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
    
    def visualize_resource_usage(self, monitoring_logs: List[Dict], save_path: Optional[str] = None):
        """Visualize CPU and memory usage during training"""
        if not monitoring_logs:
            print("No monitoring logs provided")
            return
        
        # Combine metrics from all logs, sorted by timestamp
        all_metrics = []
        for log in monitoring_logs:
            all_metrics.extend(log.get("metrics", []))
        
        all_metrics.sort(key=lambda x: x.get("timestamp", 0))
        
        # Extract data
        timestamps = [m.get("elapsed_seconds", 0) / 60 for m in all_metrics]  # Convert to minutes
        cpu_percent = [m.get("cpu_percent", 0) for m in all_metrics]
        memory_percent = [m.get("memory_percent", 0) for m in all_metrics]
        process_memory = [m.get("process_memory_gb", 0) for m in all_metrics]
        
        # Create plot with two y-axes
        fig, ax1 = plt.subplots(figsize=(12, 7))
        
        # CPU percentage (left y-axis)
        color = 'tab:blue'
        ax1.set_xlabel('Time (minutes)', fontsize=14)
        ax1.set_ylabel('CPU Usage (%)', color=color, fontsize=14)
        ax1.plot(timestamps, cpu_percent, color=color, linewidth=2, label='CPU Usage')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(0, 100)
        
        # Memory usage (right y-axis)
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Memory Usage (GB)', color=color, fontsize=14)
        ax2.plot(timestamps, process_memory, color=color, linewidth=2, label='Process Memory')
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Title and grid
        plt.title('CPU and Memory Usage During GRPO Training', fontsize=16)
        ax1.grid(True, alpha=0.3)
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # Save or display
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
    
    def generate_all_visualizations(self):
        """Generate all available visualizations"""
        # Create visualizations directory
        viz_dir = self.output_dir
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating visualizations in: {viz_dir}")
        
        # Load monitoring logs
        try:
            monitoring_logs = self.load_monitoring_logs()
            if monitoring_logs:
                print(f"Found {len(monitoring_logs)} monitoring logs")
                self.visualize_resource_usage(
                    monitoring_logs, 
                    save_path=str(viz_dir / "resource_usage.png")
                )
                print("Generated resource usage visualization")
        except Exception as e:
            print(f"Error generating resource usage visualization: {e}")
        
        # Load training metrics
        try:
            # Look for training metrics files
            metrics_files = list(self.log_dir.glob("training_metrics.json"))
            if not metrics_files:
                metrics_files = list(self.log_dir.glob("*metrics*.json"))
            
            if metrics_files:
                metrics_file = metrics_files[0]
                print(f"Found training metrics: {metrics_file}")
                
                log_data = self.load_training_logs(metrics_file.name)
                self.visualize_training_loss(
                    log_data,
                    save_path=str(viz_dir / "training_loss.png")
                )
                print("Generated training loss visualization")
        except Exception as e:
            print(f"Error generating training loss visualization: {e}")
        
        print("Visualization generation complete")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="GRPO Training Results Visualizer")
    parser.add_argument("--log-dir", type=str, default="./logs",
                        help="Directory containing training logs")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save visualizations")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    visualizer = TrainingVisualizer(args.log_dir, args.output_dir)
    visualizer.generate_all_visualizations()
