"""
Configuration utilities for the widefield transformer model.
"""

from dataclasses import dataclass
from typing import List, Optional
import logging


@dataclass
class ModelConfig:
    """Configuration for the transformer model."""
    hidden_dim: int = 128  # Reduced from 256
    num_heads: int = 4     # Reduced from 8
    num_layers: int = 3    # Reduced from 6
    ff_dim: int = 256      # Reduced from 1024
    num_classes: int = 2
    dropout: float = 0.3   # Increased from 0.1


@dataclass
class TrainingConfig:
    """Configuration for training."""
    num_epochs: int = 50
    batch_size: int = 16      # Reduced from 32 for more gradient updates
    learning_rate: float = 5e-5  # Reduced from 1e-4
    weight_decay: float = 1e-3   # Increased from 1e-5
    num_workers: int = 4
    save_dir: str = "checkpoints"
    
    # Learning rate scheduling
    scheduler_type: str = "cosine_warmup"  # "cosine_warmup", "reduce_on_plateau", "cosine", "step"
    warmup_epochs: int = 5  # Number of epochs for linear warmup
    cosine_t_0: int = 10  # Initial restart period for cosine annealing
    cosine_t_mult: int = 2  # Factor to increase restart period
    cosine_eta_min: float = 1e-6  # Minimum learning rate for cosine annealing
    
    # Cross-validation
    n_splits: int = 5
    current_fold: int = 0


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    mat_file_path: str = "/Users/josueortegacaro/Documents/tableForModeling.mat"
    data_type: str = "dff"  # "dff" or "zscore"
    
    # Task definition
    stim_values: List[int] = None
    response_values: List[int] = None
    phases: List[str] = None
    task_name: str = "stim1_response1_early_late"
    
    def __post_init__(self):
        if self.stim_values is None:
            self.stim_values = [1]  # Only stimulus 1
        if self.response_values is None:
            self.response_values = [1]  # Only lick responses
        if self.phases is None:
            self.phases = ["early", "late"]  # Default: classify early vs late
    
    def set_phase_comparison(self, *phases):
        """
        Set the phases to compare and update task name.
        
        Args:
            *phases: Phases to compare (e.g., 'early', 'mid', 'late', 'earlyrev', 'laterev', or integers 0, 1, 2)
                    Note: 'mid' and integer 1 are treated as equivalent
        """
        # Normalize phases: handle integers and strings
        # Convert string representations of integers to actual integers first
        phase_map_int = {0: 'early', 1: 'mid', 2: 'late'}
        normalized_phases = []
        for phase in phases:
            # Check if it's a string representation of an integer (e.g., "1", "0", "2")
            if isinstance(phase, str) and phase.strip().isdigit():
                # Convert string "1" to integer 1, then map to string
                phase_int = int(phase.strip())
                normalized_phases.append(phase_map_int.get(phase_int, str(phase_int).lower()))
            elif isinstance(phase, str) and phase.strip().lower() == 'int':
                # String 'int' maps to 'mid' (since integer 1 maps to 'mid')
                normalized_phases.append('mid')
            elif isinstance(phase, (int, type(1))):
                # Integer phase - map to string
                normalized_phases.append(phase_map_int.get(phase, str(phase).lower()))
            else:
                # String phase - normalize to lowercase (allows any phase value)
                normalized_phases.append(str(phase).strip().lower())
        
        if len(normalized_phases) < 2:
            raise ValueError("Must provide at least 2 phases to compare")
        
        if len(set(normalized_phases)) != len(normalized_phases):
            raise ValueError("Cannot compare duplicate phases")
        
        self.phases = normalized_phases
        phase_str = "_vs_".join(normalized_phases)
        self.task_name = f"stim{self.stim_values[0]}_response{self.response_values[0]}_{phase_str}"


@dataclass
class WandbConfig:
    """Configuration for Weights & Biases logging."""
    project: str = "widefieldmodeling"
    entity: str = "josueortc"
    enabled: bool = True
    
    
@dataclass
class AnalysisConfig:
    """Configuration for analysis and evaluation."""
    attention_samples: int = 1000
    results_dir: str = "results"
    create_plots: bool = True
    save_attention_matrix: bool = True


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    wandb: WandbConfig
    analysis: AnalysisConfig
    
    # Global settings
    seed: int = 42
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    
    def __post_init__(self):
        # Set default instances if not provided
        if not isinstance(self.model, ModelConfig):
            self.model = ModelConfig()
        if not isinstance(self.training, TrainingConfig):
            self.training = TrainingConfig()
        if not isinstance(self.data, DataConfig):
            self.data = DataConfig()
        if not isinstance(self.wandb, WandbConfig):
            self.wandb = WandbConfig()
        if not isinstance(self.analysis, AnalysisConfig):
            self.analysis = AnalysisConfig()


def create_default_config() -> ExperimentConfig:
    """Create a default experiment configuration."""
    return ExperimentConfig(
        model=ModelConfig(),
        training=TrainingConfig(),
        data=DataConfig(),
        wandb=WandbConfig(),
        analysis=AnalysisConfig()
    )


def update_model_classes_from_phases(config: ExperimentConfig) -> None:
    """
    Update model num_classes based on the phases being compared.
    
    Args:
        config: Experiment configuration to update
    """
    num_phases = len(config.data.phases)
    config.model.num_classes = num_phases
    
    logger = logging.getLogger(__name__)
    logger.info(f"Updated model num_classes to {num_phases} based on phases: {config.data.phases}")


def get_wandb_config(config: ExperimentConfig) -> dict:
    """Convert experiment config to wandb config format."""
    return {
        # Model parameters
        "hidden_dim": config.model.hidden_dim,
        "num_heads": config.model.num_heads,
        "num_layers": config.model.num_layers,
        "ff_dim": config.model.ff_dim,
        "dropout": config.model.dropout,
        
        # Training parameters
        "num_epochs": config.training.num_epochs,
        "batch_size": config.training.batch_size,
        "learning_rate": config.training.learning_rate,
        "weight_decay": config.training.weight_decay,
        
        # Scheduler parameters
        "scheduler_type": config.training.scheduler_type,
        "warmup_epochs": config.training.warmup_epochs,
        "cosine_t_0": config.training.cosine_t_0,
        "cosine_t_mult": config.training.cosine_t_mult,
        "cosine_eta_min": config.training.cosine_eta_min,
        
        # Data parameters
        "data_type": config.data.data_type,
        "task_name": config.data.task_name,
        "stim_values": config.data.stim_values,
        "response_values": config.data.response_values,
        "phases": config.data.phases,
        
        # Cross-validation
        "n_splits": config.training.n_splits,
        "current_fold": config.training.current_fold,
        
        # Other
        "seed": config.seed
    }
