#!/usr/bin/env python3
"""
GRPO Training Performance Monitor
Tracks and logs system resources during training
"""

import os
import sys
import time
import psutil
import argparse
import json
import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project utilities
from src.utils.cpu_utils import print_system_info


class TrainingMonitor:
    """Monitor and log system resources during training"""
    
    def __init__(self, output_dir: str, interval: int = 5):
        """
        Initialize the training monitor
        
        Args:
            output_dir: Directory to save monitoring logs
            interval: Sampling interval in seconds
        """
        self.output_dir = Path(output_dir)
        self.interval = interval
        self.metrics = []
        self.start_time = time.time()
        self.running = False
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Output log path
        self.log_path = self.output_dir / f"monitoring_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
    def start_monitoring(self):
        """Start monitoring system resources"""
        print(f"Starting system monitoring (interval: {self.interval}s)")
        print(f"Log will be saved to: {self.log_path}")
        
        self.running = True
        self.start_time = time.time()
        
        try:
            # Record initial system state
            self.metrics.append(self._collect_metrics())
            print_system_info()
            
            # Main monitoring loop
            while self.running:
                time.sleep(self.interval)
                metrics = self._collect_metrics()
                self.metrics.append(metrics)
                
                # Print current metrics
                print(f"CPU: {metrics['cpu_percent']}% | "
                      f"Memory: {metrics['memory_used_gb']:.2f}/{metrics['memory_total_gb']:.2f} GB | "
                      f"Process Memory: {metrics['process_memory_gb']:.2f} GB")
                
                # Periodically save metrics
                if len(self.metrics) % 10 == 0:
                    self._save_metrics()
                    
        except KeyboardInterrupt:
            print("Monitoring stopped by user")
        finally:
            self.stop_monitoring()
    
    def stop_monitoring(self):
        """Stop monitoring and save final results"""
        self.running = False
        self._save_metrics()
        print(f"Monitoring stopped after {time.time() - self.start_time:.2f} seconds")
        print(f"Metrics saved to: {self.log_path}")
    
    def _collect_metrics(self) -> Dict:
        """Collect system metrics"""
        process = psutil.Process()
        memory = psutil.virtual_memory()
        
        return {
            "timestamp": time.time(),
            "elapsed_seconds": time.time() - self.start_time,
            "cpu_percent": psutil.cpu_percent(),
            "memory_total_gb": memory.total / (1024**3),
            "memory_available_gb": memory.available / (1024**3),
            "memory_used_gb": memory.used / (1024**3),
            "memory_percent": memory.percent,
            "process_memory_gb": process.memory_info().rss / (1024**3),
            "process_cpu_percent": process.cpu_percent(),
            "swap_used_gb": psutil.swap_memory().used / (1024**3) if hasattr(psutil, 'swap_memory') else 0
        }
    
    def _save_metrics(self):
        """Save metrics to JSON file"""
        with open(self.log_path, 'w') as f:
            json.dump({
                "monitoring_start": self.start_time,
                "monitoring_end": time.time(),
                "interval_seconds": self.interval,
                "metrics": self.metrics
            }, f, indent=2)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="GRPO Training Performance Monitor")
    parser.add_argument("--output", type=str, default="./logs/monitoring", 
                        help="Directory to save monitoring logs")
    parser.add_argument("--interval", type=int, default=5,
                        help="Sampling interval in seconds")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    monitor = TrainingMonitor(args.output, args.interval)
    monitor.start_monitoring()
