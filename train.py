#!/usr/bin/env python3
"""
PRISMT: Unified Training Pipeline for Widefield and CDKL5 Data

This script provides a unified interface for training transformer models
on standardized widefield calcium imaging data, supporting both:
- Widefield data: Phase classification tasks
- CDKL5 data: Genotype classification tasks
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import time
from pathlib import Path
import sys
import logging
from typing import Tuple, List, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from data.data_loader import WidefieldDataset, TaskDefinition, WidefieldTrialDataset
from models.transformer import create_model
from training.trainer import Trainer, run_attention_and_diagnosis
from utils.helpers import setup_logging, set_seed, get_device, format_time

logger = logging.getLogger(__name__)


def detect_task_type(dataset: WidefieldDataset) -> str:
    """
    Detect task type from dataset structure.
    
    Returns:
        'genotype' for CDKL5 genotype classification
        'phase' for widefield phase classification
    """
    # Check if datasets have 'label' field (CDKL5) or phase diversity (widefield)
    has_labels = False
    has_phase_diversity = False
    phases_seen = set()
    
    for i in range(min(10, dataset.data_table.shape[0])):  # Check first 10 datasets
        if i >= dataset.data_table.shape[0]:
            break
        
        # Check for label field (CDKL5)
        try:
            phase_data = dataset.data_table[i, 4]  # phase column
            mouse_data = dataset.data_table[i, 5]  # mouse column
            
            # Check if mouse IDs indicate genotype
            if isinstance(mouse_data, str):
                mouse_lower = mouse_data.lower()
                if mouse_lower.startswith('wt_') or mouse_lower.startswith('mut_'):
                    has_labels = True
            
            # Check phase diversity
            if isinstance(phase_data, str):
                phases_seen.add(phase_data.lower())
            elif hasattr(phase_data, '__iter__') and not isinstance(phase_data, str):
                unique_phases = np.unique([str(p).lower() for p in phase_data])
                phases_seen.update(unique_phases)
        except:
            pass
    
    if has_labels or len(phases_seen) == 1 and 'all' in phases_seen:
        return 'genotype'
    elif len(phases_seen) > 1:
        return 'phase'
    else:
        # Default to phase classification
        return 'phase'


def create_data_loaders_unified(
    dataset: WidefieldDataset,
    train_indices: List[int],
    val_indices: List[int],
    task_type: str,
    data_type: str = 'dff',
    batch_size: int = 32,
    num_workers: int = 4,
    phase1: Optional[str] = None,
    phase2: Optional[str] = None
) -> Tuple[DataLoader, DataLoader, dict]:
    """
    Create data loaders for unified training pipeline.
    
    Args:
        dataset: WidefieldDataset instance
        train_indices: Training dataset indices
        val_indices: Validation dataset indices
        task_type: 'genotype' or 'phase'
        data_type: 'dff' or 'zscore'
        batch_size: Batch size
        num_workers: Number of data loader workers
        phase1: First phase for phase classification (if task_type='phase')
        phase2: Second phase for phase classification (if task_type='phase')
    
    Returns:
        Tuple of (train_loader, val_loader, normalization_stats)
    """
    if task_type == 'genotype':
        return create_genotype_data_loaders(
            dataset, train_indices, val_indices, data_type, batch_size, num_workers
        )
    else:
        return create_phase_data_loaders(
            dataset, train_indices, val_indices, data_type, batch_size, num_workers,
            phase1, phase2
        )


def create_genotype_data_loaders(
    dataset: WidefieldDataset,
    train_indices: List[int],
    val_indices: List[int],
    data_type: str = 'dff',
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, dict]:
    """Create data loaders for genotype classification (CDKL5)."""
    # Detect available stim and response values
    all_stim_values = set()
    all_response_values = set()
    
    for dataset_idx in list(train_indices) + list(val_indices):
        if dataset_idx < dataset.data_table.shape[0]:
            stim_data = dataset.data_table[dataset_idx, 2]
            response_data = dataset.data_table[dataset_idx, 3]
            
            if hasattr(stim_data, '__iter__') and not isinstance(stim_data, str):
                unique_stim = np.unique(stim_data)
                unique_stim = unique_stim[~np.isnan(unique_stim)]
                if len(unique_stim) > 0:
                    all_stim_values.update(unique_stim.astype(int))
            
            if response_data is not None:
                if hasattr(response_data, '__iter__') and not isinstance(response_data, str):
                    unique_response = np.unique(response_data)
                    unique_response = unique_response[~np.isnan(unique_response)]
                    if len(unique_response) > 0:
                        all_response_values.update(unique_response.astype(int))
    
    stim_values_list = sorted(list(all_stim_values)) if all_stim_values else [1]
    response_values_list = sorted(list(all_response_values)) if all_response_values else [1]
    
    logger.info(f"Detected stim values: {stim_values_list}")
    logger.info(f"Detected response values: {response_values_list}")
    
    # Create task definition
    task_definition = TaskDefinition(
        stim_values=stim_values_list,
        response_values=response_values_list,
        phases=['all'],
        task_name="genotype_classification"
    )
    
    # Filter and extract data
    train_neural_data, train_labels_phase, train_trial_indices, train_mouse_ids = task_definition.filter_trials(
        dataset, train_indices, data_type
    )
    val_neural_data, val_labels_phase, val_trial_indices, val_mouse_ids = task_definition.filter_trials(
        dataset, val_indices, data_type
    )
    
    # Create genotype labels from mouse IDs
    train_genotype_labels = []
    val_genotype_labels = []
    
    for mouse_id in train_mouse_ids:
        mouse_id_str = str(mouse_id).lower()
        if mouse_id_str.startswith('wt_') or mouse_id_str.startswith('wild'):
            train_genotype_labels.append(0)  # Wild type
        elif mouse_id_str.startswith('mut_') or mouse_id_str.startswith('mutant'):
            train_genotype_labels.append(1)  # Mutant
        else:
            train_genotype_labels.append(0)  # Default to WT
    
    for mouse_id in val_mouse_ids:
        mouse_id_str = str(mouse_id).lower()
        if mouse_id_str.startswith('wt_') or mouse_id_str.startswith('wild'):
            val_genotype_labels.append(0)
        elif mouse_id_str.startswith('mut_') or mouse_id_str.startswith('mutant'):
            val_genotype_labels.append(1)
        else:
            val_genotype_labels.append(0)
    
    train_genotype_labels = np.array(train_genotype_labels)
    val_genotype_labels = np.array(val_genotype_labels)
    
    logger.info(f"Train genotype distribution: WT={np.sum(train_genotype_labels==0)}, Mut={np.sum(train_genotype_labels==1)}")
    logger.info(f"Val genotype distribution: WT={np.sum(val_genotype_labels==0)}, Mut={np.sum(val_genotype_labels==1)}")
    
    # Normalization stats
    normalization_stats = {'normalization_type': 'scale_20'}
    
    # Create datasets
    train_dataset = WidefieldTrialDataset(
        neural_data=train_neural_data,
        labels=train_genotype_labels,
        normalize_stats=normalization_stats,
        mouse_ids=train_mouse_ids
    )
    val_dataset = WidefieldTrialDataset(
        neural_data=val_neural_data,
        labels=val_genotype_labels,
        normalize_stats=normalization_stats,
        mouse_ids=val_mouse_ids
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, normalization_stats


def create_phase_data_loaders(
    dataset: WidefieldDataset,
    train_indices: List[int],
    val_indices: List[int],
    data_type: str = 'dff',
    batch_size: int = 32,
    num_workers: int = 4,
    phase1: Optional[str] = None,
    phase2: Optional[str] = None
) -> Tuple[DataLoader, DataLoader, dict]:
    """Create data loaders for phase classification (widefield)."""
    if phase1 is None or phase2 is None:
        raise ValueError("phase1 and phase2 must be specified for phase classification")
    
    # Detect available stim and response values
    all_stim_values = set()
    all_response_values = set()
    
    for dataset_idx in list(train_indices) + list(val_indices):
        if dataset_idx < dataset.data_table.shape[0]:
            stim_data = dataset.data_table[dataset_idx, 2]
            response_data = dataset.data_table[dataset_idx, 3]
            
            if hasattr(stim_data, '__iter__') and not isinstance(stim_data, str):
                unique_stim = np.unique(stim_data)
                unique_stim = unique_stim[~np.isnan(unique_stim)]
                if len(unique_stim) > 0:
                    all_stim_values.update(unique_stim.astype(int))
            
            if response_data is not None:
                if hasattr(response_data, '__iter__') and not isinstance(response_data, str):
                    unique_response = np.unique(response_data)
                    unique_response = unique_response[~np.isnan(unique_response)]
                    if len(unique_response) > 0:
                        all_response_values.update(unique_response.astype(int))
    
    stim_values_list = sorted(list(all_stim_values)) if all_stim_values else [1]
    response_values_list = sorted(list(all_response_values)) if all_response_values else [0, 1]
    
    logger.info(f"Detected stim values: {stim_values_list}")
    logger.info(f"Detected response values: {response_values_list}")
    
    # Create task definition
    task_definition = TaskDefinition(
        stim_values=stim_values_list,
        response_values=response_values_list,
        phases=[phase1, phase2],
        task_name=f"{phase1}_vs_{phase2}"
    )
    
    # Filter and extract data
    train_neural_data, train_labels, train_trial_indices, train_mouse_ids = task_definition.filter_trials(
        dataset, train_indices, data_type
    )
    val_neural_data, val_labels, val_trial_indices, val_mouse_ids = task_definition.filter_trials(
        dataset, val_indices, data_type
    )
    
    logger.info(f"Train phase distribution: {np.bincount(train_labels)}")
    logger.info(f"Val phase distribution: {np.bincount(val_labels)}")
    
    # Normalization stats
    normalization_stats = {'normalization_type': 'scale_20'}
    
    # Create datasets
    train_dataset = WidefieldTrialDataset(
        neural_data=train_neural_data,
        labels=train_labels,
        normalize_stats=normalization_stats,
        mouse_ids=train_mouse_ids
    )
    val_dataset = WidefieldTrialDataset(
        neural_data=val_neural_data,
        labels=val_labels,
        normalize_stats=normalization_stats,
        mouse_ids=val_mouse_ids
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, normalization_stats


def create_train_val_split(dataset: WidefieldDataset, task_type: str, val_split: float = 0.2, seed: int = 42) -> Tuple[List[int], List[int]]:
    """Create train/validation split."""
    np.random.seed(seed)
    
    if task_type == 'genotype':
        # Group by genotype
        wt_indices = []
        mut_indices = []
        
        for i in range(dataset.data_table.shape[0]):
            mouse_id = dataset.data_table[i, 5]
            mouse_id_str = str(mouse_id).lower()
            if mouse_id_str.startswith('wt_') or mouse_id_str.startswith('wild'):
                wt_indices.append(i)
            elif mouse_id_str.startswith('mut_') or mouse_id_str.startswith('mutant'):
                mut_indices.append(i)
            else:
                n_total = dataset.data_table.shape[0]
                if i < n_total / 2:
                    wt_indices.append(i)
                else:
                    mut_indices.append(i)
        
        np.random.shuffle(wt_indices)
        np.random.shuffle(mut_indices)
        
        wt_split = int(len(wt_indices) * (1 - val_split))
        mut_split = int(len(mut_indices) * (1 - val_split))
        
        train_indices = wt_indices[:wt_split] + mut_indices[:mut_split]
        val_indices = wt_indices[wt_split:] + mut_indices[mut_split:]
    else:
        # Group by mouse for phase classification
        mouse_groups = {}
        for i in range(dataset.data_table.shape[0]):
            mouse_id = dataset.data_table[i, 5]
            if mouse_id not in mouse_groups:
                mouse_groups[mouse_id] = []
            mouse_groups[mouse_id].append(i)
        
        mice = list(mouse_groups.keys())
        np.random.shuffle(mice)
        
        n_val_mice = int(len(mice) * val_split)
        val_mice = set(mice[:n_val_mice])
        
        train_indices = []
        val_indices = []
        
        for mouse_id, indices in mouse_groups.items():
            if mouse_id in val_mice:
                val_indices.extend(indices)
            else:
                train_indices.extend(indices)
    
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    
    logger.info(f"Created split: {len(train_indices)} train, {len(val_indices)} val")
    
    return train_indices, val_indices


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='PRISMT Unified Training Pipeline')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to standardized .mat data file')
    parser.add_argument('--data_type', type=str, default='dff', choices=['dff', 'zscore'],
                        help='Type of neural data to use')
    parser.add_argument('--task_type', type=str, default='auto', choices=['auto', 'genotype', 'phase'],
                        help='Task type: auto (detect), genotype (CDKL5), or phase (widefield)')
    parser.add_argument('--phase1', type=str, default=None,
                        help='First phase for phase classification (e.g., "early")')
    parser.add_argument('--phase2', type=str, default=None,
                        help='Second phase for phase classification (e.g., "late")')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help='Weight decay')
    
    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of transformer layers')
    parser.add_argument('--ff_dim', type=int, default=256,
                        help='Feed forward dimension')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    
    # Training control
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split fraction')
    parser.add_argument('--scheduler_type', type=str, default='cosine_warmup',
                        choices=['cosine_warmup', 'cosine', 'reduce_on_plateau', 'step'],
                        help='Learning rate scheduler type')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of warmup epochs')
    parser.add_argument('--cosine_t_0', type=int, default=10,
                        help='Initial restart period for cosine annealing')
    parser.add_argument('--cosine_t_mult', type=int, default=2,
                        help='Factor to increase restart period')
    parser.add_argument('--cosine_eta_min', type=float, default=1e-6,
                        help='Minimum learning rate')
    
    # Other parameters
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--attention_samples', type=int, default=1000,
                        help='Number of samples for attention extraction (post-training)')
    
    # WandB arguments
    parser.add_argument('--wandb_project', type=str, default='prismt',
                        help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default='josueortc',
                        help='WandB entity name')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable WandB logging')
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Setup logging
    setup_logging()
    logger.info("=" * 80)
    logger.info("PRISMT: Unified Training Pipeline")
    logger.info("=" * 80)
    
    # Set seed
    set_seed(args.seed)
    
    # Get device
    device = get_device(args.device)
    
    # Load dataset
    logger.info(f"Loading dataset from: {args.data_path}")
    dataset = WidefieldDataset(args.data_path)
    logger.info(f"Loaded {dataset.data_table.shape[0]} datasets")
    
    # Detect or set task type
    if args.task_type == 'auto':
        task_type = detect_task_type(dataset)
        logger.info(f"Auto-detected task type: {task_type}")
    else:
        task_type = args.task_type
    
    # Validate phase arguments for phase classification
    if task_type == 'phase':
        if args.phase1 is None or args.phase2 is None:
            logger.error("phase1 and phase2 must be specified for phase classification")
            sys.exit(1)
    
    # Create train/val split
    train_indices, val_indices = create_train_val_split(
        dataset, task_type, args.val_split, args.seed
    )
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, normalization_stats = create_data_loaders_unified(
        dataset, train_indices, val_indices, task_type, args.data_type,
        args.batch_size, num_workers=4,
        phase1=args.phase1, phase2=args.phase2
    )
    
    # Get sample data to determine dimensions
    sample_data, _ = next(iter(train_loader))
    sample_data = sample_data[0]  # Get first sample
    n_brain_areas = sample_data.shape[1]
    time_points = sample_data.shape[0]
    
    logger.info(f"Data dimensions: {n_brain_areas} brain areas, {time_points} time points")
    
    # Create model
    logger.info("Creating model...")
    model = create_model(
        n_brain_areas=n_brain_areas,
        time_points=time_points,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        num_classes=2,
        dropout=args.dropout,
        device=device
    )
    
    model_info = model.get_model_info()
    logger.info(f"Model created: {model_info}")
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        save_dir=args.save_dir,
        wandb_project=args.wandb_project if not args.no_wandb else None,
        wandb_entity=args.wandb_entity if not args.no_wandb else None,
        scheduler_type=args.scheduler_type,
        warmup_epochs=args.warmup_epochs,
        cosine_t_0=args.cosine_t_0,
        cosine_t_mult=args.cosine_t_mult,
        cosine_eta_min=args.cosine_eta_min
    )
    
    # Train
    logger.info("Starting training...")
    start_time = time.time()
    
    wandb_config = {
        'task_type': task_type,
        'data_type': args.data_type,
        'n_brain_areas': n_brain_areas,
        'time_points': time_points,
        'hidden_dim': args.hidden_dim,
        'num_heads': args.num_heads,
        'num_layers': args.num_layers,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'phase1': args.phase1,
        'phase2': args.phase2
    }
    
    history = trainer.train(num_epochs=args.epochs, config=wandb_config)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Training completed in {format_time(elapsed_time)}")
    
    # Final validation
    logger.info("Running final validation...")
    val_loss, val_acc, val_f1 = trainer.validate_epoch()
    logger.info(f"Final validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}")
    
    # Attention extraction and diagnosis on best model
    logger.info("Extracting attention and running diagnosis on best model...")
    best_model = trainer.get_best_model()
    run_attention_and_diagnosis(
        model=best_model,
        data_loader=val_loader,
        device=device,
        save_dir=args.save_dir,
        num_samples=args.attention_samples,
        num_classes=2,
    )
    
    logger.info("=" * 80)
    logger.info("Training pipeline completed successfully!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
