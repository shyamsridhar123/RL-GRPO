import os
import gc
import torch
import logging
import psutil

def setup_cpu_optimization(num_physical_cores=12, num_logical_cores=14):
    """
    Configure the CPU environment for optimal training performance
    
    Args:
        num_physical_cores: Number of physical cores to use for OpenMP
        num_logical_cores: Number of logical cores to use for PyTorch
    """
    # Set OpenMP threads to physical core count
    os.environ['OMP_NUM_THREADS'] = str(num_physical_cores)
    
    # Set PyTorch threads to logical core count
    torch.set_num_threads(num_logical_cores)
    
    # Disable CUDA if available to ensure CPU-only operation
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    logging.info(f"CPU optimization configured: OMP_NUM_THREADS={num_physical_cores}, "
                 f"PyTorch threads={num_logical_cores}")

def optimize_memory():
    """
    Perform garbage collection and memory optimization
    
    Returns:
        Current memory usage in GB
    """
    # Force garbage collection
    gc.collect()
    
    # Clear CUDA cache if available (defensive)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Get current memory usage
    memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3
    
    return memory_usage

def print_system_info():
    """Print current system information for debugging"""
    mem = psutil.virtual_memory()
    cpu_count = psutil.cpu_count(logical=False)
    logical_cpus = psutil.cpu_count(logical=True)
    
    print("=== System Information ===")
    print(f"CPU: {cpu_count} physical cores, {logical_cpus} logical cores")
    print(f"Memory: {mem.total / (1024**3):.2f}GB total, {mem.available / (1024**3):.2f}GB available")
    print(f"Current process memory: {psutil.Process(os.getpid()).memory_info().rss / (1024**3):.2f}GB")
    print(f"PyTorch threads: {torch.get_num_threads()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print("==========================")
