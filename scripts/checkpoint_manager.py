#!/usr/bin/env python3
"""
GRPO Model Checkpoint Manager
Handles model checkpoint operations, cleanup, and optimization
"""

import os
import sys
import shutil
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ModelCheckpointManager:
    """
    Manages model checkpoints for GRPO training
    - Organizes checkpoints
    - Cleans up old/redundant checkpoints
    - Optimizes checkpoints for deployment
    """
    
    def __init__(self, models_dir: str):
        """
        Initialize the checkpoint manager
        
        Args:
            models_dir: Directory containing model checkpoints
        """
        self.models_dir = Path(models_dir)
        
    def list_checkpoints(self, stage: Optional[str] = None) -> List[Dict]:
        """
        List all checkpoints
        
        Args:
            stage: Optional stage filter (e.g., 'stage1', 'stage2', etc.)
            
        Returns:
            List of checkpoint info dictionaries
        """
        checkpoints = []
        
        # Determine directories to scan
        if stage:
            dirs_to_scan = [self.models_dir / stage] if (self.models_dir / stage).exists() else []
        else:
            # List all subdirectories that might contain models
            dirs_to_scan = [d for d in self.models_dir.iterdir() if d.is_dir()]
        
        # Scan each directory for checkpoint files
        for model_dir in dirs_to_scan:
            # Check if this is a valid model directory
            config_file = model_dir / "config.json"
            model_file = model_dir / "pytorch_model.bin"
            safetensors_file = model_dir / "model.safetensors"
            
            is_valid = ((config_file.exists() and model_file.exists()) or 
                        (config_file.exists() and safetensors_file.exists()))
            
            if not is_valid:
                # Check for final_model subdirectory
                final_model_dir = model_dir / "final_model"
                if final_model_dir.exists():
                    # Recursively check the final_model directory
                    config_file = final_model_dir / "config.json"
                    model_file = final_model_dir / "pytorch_model.bin"
                    safetensors_file = final_model_dir / "model.safetensors"
                    is_valid = ((config_file.exists() and model_file.exists()) or 
                                (config_file.exists() and safetensors_file.exists()))
                    if is_valid:
                        model_dir = final_model_dir
            
            if is_valid:
                # Get model size
                if model_file.exists():
                    size_mb = model_file.stat().st_size / (1024 * 1024)
                    model_format = "pytorch_model"
                else:
                    size_mb = safetensors_file.stat().st_size / (1024 * 1024)
                    model_format = "safetensors"
                
                # Get last modified time
                last_modified = max(
                    config_file.stat().st_mtime,
                    model_file.stat().st_mtime if model_file.exists() else 0,
                    safetensors_file.stat().st_mtime if safetensors_file.exists() else 0
                )
                
                # Get metadata if available
                metadata = {}
                metadata_file = model_dir / "training_metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                    except Exception:
                        pass
                
                checkpoints.append({
                    "path": str(model_dir),
                    "name": model_dir.name,
                    "stage": model_dir.parent.name if "stage" in model_dir.parent.name else None,
                    "size_mb": size_mb,
                    "format": model_format,
                    "last_modified": datetime.datetime.fromtimestamp(last_modified).strftime('%Y-%m-%d %H:%M:%S'),
                    "metadata": metadata
                })
        
        # Sort by last modified date (newest first)
        checkpoints.sort(key=lambda x: x["last_modified"], reverse=True)
        return checkpoints
    
    def create_backup(self, checkpoint_path: str) -> str:
        """
        Create a backup of a checkpoint
        
        Args:
            checkpoint_path: Path to the checkpoint
            
        Returns:
            Path to the backup
        """
        src_path = Path(checkpoint_path)
        if not src_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Create backup in the archive directory
        archive_dir = self.models_dir / "archive"
        archive_dir.mkdir(exist_ok=True)
        
        # Create timestamped backup directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{src_path.name}_backup_{timestamp}"
        backup_path = archive_dir / backup_name
        
        print(f"Creating backup: {src_path} -> {backup_path}")
        
        # Copy the model directory
        shutil.copytree(src_path, backup_path)
        
        return str(backup_path)
    
    def cleanup_checkpoints(self, keep_latest: int = 3, dry_run: bool = True) -> List[str]:
        """
        Clean up old checkpoints, keeping only the specified number of latest ones
        
        Args:
            keep_latest: Number of latest checkpoints to keep for each stage
            dry_run: If True, only simulate the cleanup without deleting files
            
        Returns:
            List of checkpoints that would be/were removed
        """
        # Group checkpoints by stage
        checkpoints_by_stage = {}
        all_checkpoints = self.list_checkpoints()
        
        for checkpoint in all_checkpoints:
            stage = checkpoint.get("stage") or "unknown"
            if stage not in checkpoints_by_stage:
                checkpoints_by_stage[stage] = []
            checkpoints_by_stage[stage].append(checkpoint)
        
        # For each stage, keep only the latest N checkpoints
        to_remove = []
        
        for stage, checkpoints in checkpoints_by_stage.items():
            # Sort by last modified (newest first)
            sorted_checkpoints = sorted(checkpoints, key=lambda x: x["last_modified"], reverse=True)
            
            # Keep the latest N
            if len(sorted_checkpoints) > keep_latest:
                to_remove.extend(sorted_checkpoints[keep_latest:])
        
        if dry_run:
            print(f"DRY RUN: Would remove {len(to_remove)} checkpoints")
            for checkpoint in to_remove:
                print(f"  - {checkpoint['path']} ({checkpoint['size_mb']:.2f} MB)")
        else:
            print(f"Removing {len(to_remove)} checkpoints")
            for checkpoint in to_remove:
                try:
                    checkpoint_path = Path(checkpoint['path'])
                    print(f"  - Removing {checkpoint_path} ({checkpoint['size_mb']:.2f} MB)")
                    
                    # Check if this is the final_model subdirectory
                    if checkpoint_path.name == "final_model":
                        # Delete the parent directory instead
                        shutil.rmtree(checkpoint_path.parent)
                    else:
                        shutil.rmtree(checkpoint_path)
                except Exception as e:
                    print(f"    Error removing {checkpoint_path}: {e}")
        
        return [c['path'] for c in to_remove]
    
    def convert_to_safetensors(self, checkpoint_path: str) -> bool:
        """
        Convert PyTorch model to safetensors format for better compatibility
        
        Args:
            checkpoint_path: Path to the checkpoint
            
        Returns:
            True if conversion was successful
        """
        try:
            # Lazy import to avoid dependency issues
            from safetensors.torch import save_file as save_safetensors
            import torch
            
            checkpoint_path = Path(checkpoint_path)
            model_file = checkpoint_path / "pytorch_model.bin"
            safetensors_file = checkpoint_path / "model.safetensors"
            
            if not model_file.exists():
                print(f"PyTorch model file not found: {model_file}")
                return False
            
            if safetensors_file.exists():
                print(f"SafeTensors file already exists: {safetensors_file}")
                return True
            
            print(f"Converting {model_file} to SafeTensors format")
            
            # Load PyTorch model
            state_dict = torch.load(model_file, map_location="cpu")
            
            # Save as SafeTensors
            save_safetensors(state_dict, safetensors_file)
            
            print(f"Conversion successful: {safetensors_file}")
            return True
            
        except ImportError:
            print("SafeTensors package not available. Install with: pip install safetensors")
            return False
        except Exception as e:
            print(f"Error converting to SafeTensors: {e}")
            return False


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="GRPO Model Checkpoint Manager")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all checkpoints")
    list_parser.add_argument("--stage", type=str, help="Filter by stage")
    
    # Backup command
    backup_parser = subparsers.add_parser("backup", help="Create a backup of a checkpoint")
    backup_parser.add_argument("checkpoint_path", type=str, help="Path to the checkpoint")
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up old checkpoints")
    cleanup_parser.add_argument("--keep", type=int, default=3, help="Number of latest checkpoints to keep for each stage")
    cleanup_parser.add_argument("--execute", action="store_true", help="Actually perform the cleanup (default is dry run)")
    
    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert a checkpoint to SafeTensors format")
    convert_parser.add_argument("checkpoint_path", type=str, help="Path to the checkpoint")
    
    # Common arguments
    parser.add_argument("--models-dir", type=str, default="./models",
                        help="Directory containing model checkpoints")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    manager = ModelCheckpointManager(args.models_dir)
    
    if args.command == "list":
        checkpoints = manager.list_checkpoints(args.stage)
        print(f"Found {len(checkpoints)} checkpoints:")
        
        for i, checkpoint in enumerate(checkpoints):
            print(f"{i+1}. {checkpoint['name']}")
            print(f"   Path: {checkpoint['path']}")
            print(f"   Size: {checkpoint['size_mb']:.2f} MB")
            print(f"   Format: {checkpoint['format']}")
            print(f"   Last Modified: {checkpoint['last_modified']}")
            if checkpoint['metadata']:
                print(f"   Training Time: {checkpoint['metadata'].get('training_time_seconds', 'N/A'):.2f}s")
            print("")
    
    elif args.command == "backup":
        backup_path = manager.create_backup(args.checkpoint_path)
        print(f"Backup created: {backup_path}")
    
    elif args.command == "cleanup":
        removed = manager.cleanup_checkpoints(args.keep, not args.execute)
        if args.execute:
            print(f"Removed {len(removed)} checkpoints")
        else:
            print(f"Would remove {len(removed)} checkpoints (dry run)")
    
    elif args.command == "convert":
        success = manager.convert_to_safetensors(args.checkpoint_path)
        if success:
            print(f"Successfully converted {args.checkpoint_path} to SafeTensors format")
        else:
            print(f"Failed to convert {args.checkpoint_path}")
    
    else:
        print("No command specified. Use --help for usage information.")
