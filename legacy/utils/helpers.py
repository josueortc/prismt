"""
Helper utilities for the widefield transformer model.
"""

import logging
import random
from pathlib import Path
import numpy as np
import torch


def setup_logging(log_level: str = "INFO") -> None:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: str = "auto") -> torch.device:
    """
    Get the appropriate device for computation.
    
    Args:
        device_str: Device specification ("auto", "cpu", "cuda", "mps")
        
    Returns:
        PyTorch device
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    
    logging.info(f"Using device: {device}")
    return device


def create_directories(*paths: str) -> None:
    """
    Create directories if they don't exist.
    
    Args:
        *paths: Variable number of directory paths to create
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def count_parameters(model: torch.nn.Module) -> tuple:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def save_experiment_info(config, save_path: str) -> None:
    """
    Save experiment configuration and info.
    
    Args:
        config: Experiment configuration
        save_path: Path to save the info
    """
    import json
    from dataclasses import asdict
    
    # Convert config to dictionary
    config_dict = asdict(config)
    
    # Save as JSON
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    logging.info(f"Saved experiment configuration to {save_path}")


def format_time(seconds: float) -> str:
    """
    Format time in seconds to a readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes}m {seconds:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours}h {minutes}m {seconds:.1f}s"


def check_data_file(file_path: str) -> bool:
    """
    Check if data file exists and is accessible.
    
    Args:
        file_path: Path to data file
        
    Returns:
        True if file exists and is accessible
    """
    path = Path(file_path)
    if not path.exists():
        logging.error(f"Data file not found: {file_path}")
        return False
    
    if not path.is_file():
        logging.error(f"Path is not a file: {file_path}")
        return False
    
    # Check if it's a .mat file
    if not path.suffix.lower() == '.mat':
        logging.warning(f"File does not have .mat extension: {file_path}")
    
    logging.info(f"Data file found: {file_path}")
    return True


def get_memory_usage() -> str:
    """
    Get current memory usage information.
    
    Returns:
        Memory usage string
    """
    import psutil
    
    # Get system memory
    memory = psutil.virtual_memory()
    memory_gb = memory.used / (1024**3)
    memory_percent = memory.percent
    
    # Get GPU memory if available
    gpu_info = ""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / (1024**3)
        gpu_info = f", GPU: {gpu_memory:.1f}GB"
    
    return f"RAM: {memory_gb:.1f}GB ({memory_percent:.1f}%){gpu_info}"


def print_model_summary(model: torch.nn.Module, input_shape: tuple) -> None:
    """
    Print a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_shape: Shape of input tensor (without batch dimension)
    """
    try:
        from torchsummary import summary
        summary(model, input_shape)
    except ImportError:
        # Fallback to basic info if torchsummary not available
        total_params, trainable_params = count_parameters(model)
        print(f"\nModel Summary:")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Input shape: {input_shape}")


class EarlyStopping:
    """
    Early stopping utility to stop training when validation metric stops improving.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'max'):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics that should be maximized, 'min' for minimized
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Check if training should be stopped.
        
        Args:
            score: Current validation score
            
        Returns:
            True if training should be stopped
        """
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop
    
    def _is_improvement(self, score: float) -> bool:
        """Check if current score is an improvement."""
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        else:
            return score < self.best_score - self.min_delta
