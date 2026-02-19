"""
Training utilities for the widefield transformer model.
"""

import json
import logging
import os
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

from models.transformer import WidefieldTransformer

logger = logging.getLogger(__name__)


class CosineAnnealingWarmupRestarts:
    """
    Custom scheduler that combines linear warmup with cosine annealing warm restarts.
    """
    
    def __init__(
        self,
        optimizer,
        warmup_epochs: int = 5,
        max_lr: float = 1e-3,
        min_lr: float = 1e-6,
        t_0: int = 10,
        t_mult: int = 2
    ):
        """
        Initialize the scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            warmup_epochs: Number of epochs for linear warmup
            max_lr: Maximum learning rate after warmup
            min_lr: Minimum learning rate in cosine annealing
            t_0: Initial restart period
            t_mult: Factor to increase restart period
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.t_0 = t_0
        self.t_mult = t_mult
        
        self.current_epoch = 0
        self.restart_epoch = warmup_epochs  # First restart after warmup
        self.current_t = t_0
        
        # Store initial lr
        self.base_lr = optimizer.param_groups[0]['lr']
        
    def step(self):
        """Update learning rate for current epoch."""
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup phase
            lr = self.base_lr + (self.max_lr - self.base_lr) * (self.current_epoch / self.warmup_epochs)
        else:
            # Cosine annealing with warm restarts
            epochs_since_restart = self.current_epoch - self.restart_epoch
            
            if epochs_since_restart >= self.current_t:
                # Time for restart
                self.restart_epoch = self.current_epoch
                self.current_t *= self.t_mult
                epochs_since_restart = 0
            
            # Cosine annealing formula
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (
                1 + math.cos(math.pi * epochs_since_restart / self.current_t)
            )
        
        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_last_lr(self):
        """Get the last learning rate."""
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self):
        """Return the state of the scheduler."""
        return {
            'warmup_epochs': self.warmup_epochs,
            'max_lr': self.max_lr,
            'min_lr': self.min_lr,
            't_0': self.t_0,
            't_mult': self.t_mult,
            'current_epoch': self.current_epoch,
            'restart_epoch': self.restart_epoch,
            'current_t': self.current_t,
            'base_lr': self.base_lr
        }
    
    def load_state_dict(self, state_dict):
        """Load the state of the scheduler."""
        self.warmup_epochs = state_dict['warmup_epochs']
        self.max_lr = state_dict['max_lr']
        self.min_lr = state_dict['min_lr']
        self.t_0 = state_dict['t_0']
        self.t_mult = state_dict['t_mult']
        self.current_epoch = state_dict['current_epoch']
        self.restart_epoch = state_dict['restart_epoch']
        self.current_t = state_dict['current_t']
        self.base_lr = state_dict['base_lr']


class Trainer:
    """
    Trainer class for the widefield transformer model.
    """
    
    def __init__(
        self,
        model: WidefieldTransformer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        save_dir: str = "checkpoints",
        wandb_project: str = "widefieldmodeling",
        wandb_entity: str = "josueortc",
        scheduler_type: str = "cosine_warmup",
        warmup_epochs: int = 5,
        cosine_t_0: int = 10,
        cosine_t_mult: int = 2,
        cosine_eta_min: float = 1e-6
    ):
        """
        Initialize the trainer.
        
        Args:
            model: WidefieldTransformer model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            save_dir: Directory to save checkpoints
            wandb_project: Weights & Biases project name
            wandb_entity: Weights & Biases entity/user name
            scheduler_type: Type of scheduler ("cosine_warmup", "reduce_on_plateau", "cosine", "step")
            warmup_epochs: Number of epochs for linear warmup
            cosine_t_0: Initial restart period for cosine annealing
            cosine_t_mult: Factor to increase restart period
            cosine_eta_min: Minimum learning rate for cosine annealing
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Optimizer and loss function
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        # Use label smoothing to reduce overfitting
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Learning rate scheduler
        self.scheduler_type = scheduler_type
        if scheduler_type == "cosine_warmup":
            self.scheduler = CosineAnnealingWarmupRestarts(
                self.optimizer,
                warmup_epochs=warmup_epochs,
                max_lr=learning_rate,
                min_lr=cosine_eta_min,
                t_0=cosine_t_0,
                t_mult=cosine_t_mult
            )
        elif scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=cosine_t_0,
                T_mult=cosine_t_mult,
                eta_min=cosine_eta_min
            )
        elif scheduler_type == "reduce_on_plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True
            )
        elif scheduler_type == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.5
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
        logger.info(f"Using {scheduler_type} learning rate scheduler")
        
        # Checkpointing
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.best_val_acc = 0.0
        self.best_model_path = None
        
        # Training history
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_f1_scores = []
        self.current_epoch = 0
        
        # Early stopping based on validation loss
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.early_stopping_patience = 15
        
        # Weights & Biases
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        
    def initialize_wandb(self, config: Dict) -> None:
        """
        Initialize Weights & Biases logging.
        
        Args:
            config: Configuration dictionary for wandb
        """
        # Check if wandb is already initialized
        if wandb.run is not None:
            logger.warning("WandB run already exists, finishing previous run")
            wandb.finish()
        
        # Create unique run name with fold information
        run_name = None
        if 'fold' in config:
            fold_num = config['fold']
            task_name = config.get('task_name', 'unknown_task')
            run_name = f"{task_name}_fold_{fold_num}"
        
        wandb.init(
            project=self.wandb_project,
            entity=self.wandb_entity,
            config=config,
            name=run_name,
            reinit=True  # Allow reinitializing wandb
        )
        wandb.watch(self.model, log="all", log_freq=100)
        
        logger.info(f"Initialized WandB run: {wandb.run.name}")
        
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Handle both old format (data, label) and new format (data, label, mouse_id)
            if len(batch) == 2:
                data, labels = batch
            elif len(batch) == 3:
                data, labels, _ = batch  # Ignore mouse_id during training
            else:
                raise ValueError(f"Unexpected batch format: {len(batch)} elements")
            # Move data to device
            data = data.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits, _ = self.model(data)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (more aggressive to prevent overfitting)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            # Log batch statistics
            if batch_idx % 50 == 0:
                batch_acc = (predictions == labels).float().mean().item()
                
                # Calculate batch F1 score
                batch_predictions = predictions.cpu().numpy()
                batch_labels = labels.cpu().numpy()
                
                # Calculate F1 for this batch
                tp = np.sum((batch_predictions == 1) & (batch_labels == 1))
                fp = np.sum((batch_predictions == 1) & (batch_labels == 0))
                fn = np.sum((batch_predictions == 0) & (batch_labels == 1))
                
                batch_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                batch_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                batch_f1 = 2 * (batch_precision * batch_recall) / (batch_precision + batch_recall) if (batch_precision + batch_recall) > 0 else 0.0
                
                # Also calculate macro F1 (average of both classes)
                # F1 for class 0
                tp_0 = np.sum((batch_predictions == 0) & (batch_labels == 0))
                fp_0 = np.sum((batch_predictions == 0) & (batch_labels == 1))
                fn_0 = np.sum((batch_predictions == 1) & (batch_labels == 0))
                
                precision_0 = tp_0 / (tp_0 + fp_0) if (tp_0 + fp_0) > 0 else 0.0
                recall_0 = tp_0 / (tp_0 + fn_0) if (tp_0 + fn_0) > 0 else 0.0
                f1_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0.0
                
                # F1 for class 1 (same as batch_f1 above)
                f1_1 = batch_f1
                
                batch_macro_f1 = (f1_0 + f1_1) / 2
                
                logger.info(
                    f"Train Batch {batch_idx}/{len(self.train_loader)}: "
                    f"Loss={loss.item():.4f}, Acc={batch_acc:.4f}, "
                    f"F1={batch_f1:.4f}, MacroF1={batch_macro_f1:.4f}"
                )
                
                # Log to wandb
                if wandb.run is not None:
                    wandb.log({
                        "batch_train_loss": loss.item(),
                        "batch_train_accuracy": batch_acc,
                        "batch_train_f1": batch_f1,
                        "batch_train_macro_f1": batch_macro_f1,
                        "learning_rate": self.optimizer.param_groups[0]['lr']
                    })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def validate_epoch(self) -> Tuple[float, float, float]:
        """
        Validate for one epoch.
        
        Returns:
            Tuple of (average_loss, accuracy, f1_score)
        """
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # For F1 score calculation and detailed analysis
        all_predictions = []
        all_labels = []
        all_logits = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Handle both old format (data, label) and new format (data, label, mouse_id)
                if len(batch) == 2:
                    data, labels = batch
                elif len(batch) == 3:
                    data, labels, _ = batch  # Ignore mouse_id during validation
                else:
                    raise ValueError(f"Unexpected batch format: {len(batch)} elements")
                # Move data to device
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits, _ = self.model(data)
                loss = self.criterion(logits, labels)
                
                # Statistics
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_samples += labels.size(0)
                
                # Collect predictions, labels, and logits for detailed analysis
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_logits.extend(logits.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct_predictions / total_samples
        
        # Convert to numpy arrays for analysis
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_logits = np.array(all_logits)
        
        # Detailed prediction analysis
        unique_preds, pred_counts = np.unique(all_predictions, return_counts=True)
        unique_labels, label_counts = np.unique(all_labels, return_counts=True)
        
        # Log prediction statistics every 10 epochs for debugging
        if hasattr(self, 'current_epoch') and self.current_epoch % 10 == 0:
            logger.info(f"Validation Analysis (Epoch {self.current_epoch}):")
            logger.info(f"  Predictions distribution: {dict(zip(unique_preds, pred_counts))}")
            logger.info(f"  Labels distribution: {dict(zip(unique_labels, label_counts))}")
            logger.info(f"  Logits mean: {np.mean(all_logits, axis=0)}")
            logger.info(f"  Logits std: {np.std(all_logits, axis=0)}")
            logger.info(f"  Prediction entropy: {-np.mean(np.sum(np.exp(all_logits) / np.sum(np.exp(all_logits), axis=1, keepdims=True) * np.log(np.exp(all_logits) / np.sum(np.exp(all_logits), axis=1, keepdims=True) + 1e-8), axis=1)):.4f}")
        
        # Calculate confusion matrix elements
        tp = np.sum((all_predictions == 1) & (all_labels == 1))
        fp = np.sum((all_predictions == 1) & (all_labels == 0))
        fn = np.sum((all_predictions == 0) & (all_labels == 1))
        tn = np.sum((all_predictions == 0) & (all_labels == 0))
        
        # Compute F1 score for class 1 (positive class)
        precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0.0
        
        # Compute F1 score for class 0 (negative class)
        precision_0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0.0
        
        # Macro-averaged F1 (average of both classes)
        macro_f1 = (f1_0 + f1_1) / 2
        
        # Check for data distribution issues
        class_0_count = np.sum(all_labels == 0)
        class_1_count = np.sum(all_labels == 1)
        
        # Warning for severe class imbalance or missing classes
        if class_0_count == 0 or class_1_count == 0:
            logger.warning(f"⚠️  VALIDATION DATA ISSUE: Missing class in validation set!")
            logger.warning(f"   Class 0 samples: {class_0_count}, Class 1 samples: {class_1_count}")
            logger.warning(f"   This will cause F1 score issues. Check cross-validation splits!")
        
        # Additional logging for debugging accuracy issues
        if hasattr(self, 'current_epoch') and self.current_epoch % 10 == 0:
            logger.info(f"  Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
            logger.info(f"  Class distribution: Class0={class_0_count}, Class1={class_1_count}")
            logger.info(f"  Class 0: Precision={precision_0:.4f}, Recall={recall_0:.4f}, F1={f1_0:.4f}")
            logger.info(f"  Class 1: Precision={precision_1:.4f}, Recall={recall_1:.4f}, F1={f1_1:.4f}")
            logger.info(f"  Macro F1={macro_f1:.4f}, Binary F1={f1_1:.4f}")
            logger.info(f"  Accuracy calculation: {correct_predictions}/{total_samples} = {accuracy:.4f}")
        
        # Return macro F1 as the primary F1 score
        return avg_loss, accuracy, macro_f1
    
    def save_checkpoint(self, epoch: int, val_acc: float, is_best: bool = False) -> str:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            val_acc: Validation accuracy
            is_best: Whether this is the best model so far
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_accuracy': val_acc,
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'val_f1_scores': self.val_f1_scores
        }
        
        # Save scheduler state if available
        if hasattr(self.scheduler, 'state_dict'):
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        else:
            logger.warning(f"Scheduler {type(self.scheduler).__name__} does not support state_dict(), skipping scheduler state save")
        
        # Save regular checkpoint
        checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.best_model_path = str(best_path)
            logger.info(f"Saved new best model with validation accuracy: {val_acc:.4f}")
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if available
        if 'scheduler_state_dict' in checkpoint and hasattr(self.scheduler, 'load_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            logger.warning(f"Scheduler state not found in checkpoint or scheduler {type(self.scheduler).__name__} does not support load_state_dict()")
        
        # Restore training history
        self.train_losses = checkpoint.get('train_losses', [])
        self.train_accuracies = checkpoint.get('train_accuracies', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
        self.val_f1_scores = checkpoint.get('val_f1_scores', [])
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint
    
    def train(self, num_epochs: int, config: Optional[Dict] = None) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            config: Configuration dictionary for wandb
            
        Returns:
            Dictionary with training history
        """
        # Initialize wandb if config provided
        if config is not None:
            self.initialize_wandb(config)
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validate
            val_loss, val_acc, val_f1 = self.validate_epoch()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            self.val_f1_scores.append(val_f1)
            
            # Update learning rate scheduler
            if self.scheduler_type == "cosine_warmup":
                # Custom scheduler doesn't need validation accuracy
                current_lr = self.scheduler.step()
            elif self.scheduler_type == "reduce_on_plateau":
                # ReduceLROnPlateau needs validation accuracy
                self.scheduler.step(val_acc)
                current_lr = self.scheduler.optimizer.param_groups[0]['lr']
            else:
                # Other schedulers (cosine, step) don't need validation accuracy
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
            
            # Log epoch results
            logger.info(
                f"Epoch {epoch + 1} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                f"Val F1: {val_f1:.4f}"
            )
            
            # Log to wandb
            if wandb.run is not None:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "val_f1_score": val_f1,  # This is now macro F1
                    "val_macro_f1": val_f1,  # Explicit macro F1 logging
                    "learning_rate": current_lr
                })
            
            # Save checkpoint
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
            
            self.save_checkpoint(epoch + 1, val_acc, is_best)
            
            # Early stopping based on validation loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs (patience: {self.early_stopping_patience})")
                logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
                break
            
            # Additional check for severe overfitting
            if epoch > 10 and train_acc > 0.95 and val_acc < 0.6:
                logger.warning(f"Severe overfitting detected at epoch {epoch + 1}: train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")
                logger.warning("Consider reducing model complexity or increasing regularization")
                # Don't break, but warn the user
        
        # Final logging
        logger.info(f"\nTraining completed!")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.4f}")
        logger.info(f"Best model saved at: {self.best_model_path}")
        
        # Close wandb run
        if wandb.run is not None:
            wandb.finish()
        
        return {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'val_f1_scores': self.val_f1_scores
        }
    
    def get_best_model(self) -> WidefieldTransformer:
        """
        Load and return the best model.
        
        Returns:
            Best model with loaded weights
        """
        if self.best_model_path is None:
            logger.warning("No best model found, returning current model")
            return self.model
        
        # Create a new model instance with the same architecture as the original
        best_model = WidefieldTransformer(
            n_brain_areas=self.model.n_brain_areas,
            time_points=self.model.time_points,
            hidden_dim=self.model.hidden_dim,
            num_heads=self.model.num_heads,
            num_layers=self.model.num_layers,
            ff_dim=self.model.ff_dim,
            num_classes=self.model.num_classes,
            dropout=self.model.dropout_prob
        ).to(self.device)
        
        # Load best weights
        checkpoint = torch.load(self.best_model_path, map_location=self.device)
        best_model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Loaded best model with validation accuracy: {checkpoint['val_accuracy']:.4f}")
        return best_model


def compute_accuracy(model: WidefieldTransformer, data_loader: DataLoader, device: torch.device) -> float:
    """
    Compute accuracy on a dataset.
    
    Args:
        model: Model to evaluate
        data_loader: Data loader for evaluation
        device: Device to run evaluation on
        
    Returns:
        Accuracy as a float
    """
    model.eval()
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in data_loader:
            # Handle both old format (data, label) and new format (data, label, mouse_id)
            if len(batch) == 2:
                data, labels = batch
            elif len(batch) == 3:
                data, labels, _ = batch  # Ignore mouse_id during accuracy computation
            else:
                raise ValueError(f"Unexpected batch format: {len(batch)} elements")
            data = data.to(device)
            labels = labels.to(device)
            
            logits, _ = model(data)
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
    
    accuracy = correct_predictions / total_samples
    return accuracy


def extract_attention_rollout(
    model: WidefieldTransformer,
    data_loader: DataLoader,
    device: torch.device,
    num_samples: int = 1000,
    save_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract attention rollout from CLS token to brain area tokens, along with animal IDs.
    
    Args:
        model: Trained model
        data_loader: Data loader for evaluation
        device: Device to run on
        num_samples: Number of samples to process
        save_path: Optional path to save attention matrix and animal IDs
        
    Returns:
        Tuple of (attention_matrix, animal_ids)
        attention_matrix: shape (num_samples, n_brain_areas)
        animal_ids: shape (num_samples,) - array of animal/mouse IDs
    """
    # Unwatch model from WandB to prevent hook errors after wandb.finish()
    # WandB hooks can cause AttributeError if wandb.run is None
    try:
        import wandb
        # Try to unwatch first
        try:
            wandb.unwatch(model)
            logger.debug("Unwatched model from WandB")
        except:
            pass
        
        # Remove all hooks manually - more aggressive approach
        # WandB hooks can be stored in _forward_hooks, _forward_pre_hooks, or _backward_hooks
        hooks_removed = 0
        for name, module in model.named_modules():
            # Remove forward hooks
            if hasattr(module, '_forward_hooks') and module._forward_hooks:
                hooks_to_remove = []
                for hook_id, hook in list(module._forward_hooks.items()):
                    # Check if this is a wandb hook by examining the hook function
                    hook_str = str(hook)
                    if 'wandb' in hook_str.lower() or 'parameter_log_hook' in hook_str:
                        hooks_to_remove.append(hook_id)
                for hook_id in hooks_to_remove:
                    module._forward_hooks.pop(hook_id)
                    hooks_removed += 1
            
            # Remove forward pre-hooks
            if hasattr(module, '_forward_pre_hooks') and module._forward_pre_hooks:
                hooks_to_remove = []
                for hook_id, hook in list(module._forward_pre_hooks.items()):
                    hook_str = str(hook)
                    if 'wandb' in hook_str.lower() or 'parameter_log_hook' in hook_str:
                        hooks_to_remove.append(hook_id)
                for hook_id in hooks_to_remove:
                    module._forward_pre_hooks.pop(hook_id)
                    hooks_removed += 1
            
            # Remove backward hooks
            if hasattr(module, '_backward_hooks') and module._backward_hooks:
                hooks_to_remove = []
                for hook_id, hook in list(module._backward_hooks.items()):
                    hook_str = str(hook)
                    if 'wandb' in hook_str.lower():
                        hooks_to_remove.append(hook_id)
                for hook_id in hooks_to_remove:
                    module._backward_hooks.pop(hook_id)
                    hooks_removed += 1
        
        if hooks_removed > 0:
            logger.debug(f"Removed {hooks_removed} WandB hooks manually")
    except Exception as e:
        logger.warning(f"Could not remove WandB hooks: {e}. Continuing anyway.")
    
    model.eval()
    attention_matrices = []
    animal_ids_list = []
    samples_processed = 0
    
    with torch.no_grad():
        for batch in data_loader:
            if samples_processed >= num_samples:
                break
            
            # Handle both old format (data, label) and new format (data, label, mouse_id)
            if len(batch) == 2:
                data, _ = batch
                mouse_ids = ["unknown"] * data.size(0)
            elif len(batch) == 3:
                data, _, mouse_ids = batch
            else:
                raise ValueError(f"Unexpected batch format: {len(batch)} elements")
                
            data = data.to(device)
            batch_size = data.size(0)
            
            # Forward pass to get attention weights
            # Temporarily disable all hooks to prevent WandB errors
            # Save hooks, remove them, run forward, then restore
            saved_hooks = {}
            modules_dict = {name: module for name, module in model.named_modules()}
            for name, module in modules_dict.items():
                if hasattr(module, '_forward_hooks') and module._forward_hooks:
                    saved_hooks[name] = dict(module._forward_hooks)
                    module._forward_hooks.clear()
            
            try:
                logits, attention_weights_list = model(data)
            finally:
                # Restore hooks (though we don't need them anymore)
                for name, hooks in saved_hooks.items():
                    if name in modules_dict:
                        module = modules_dict[name]
                        if hasattr(module, '_forward_hooks'):
                            module._forward_hooks.update(hooks)
            
            # Compute attention rollout
            cls_attention = model.get_attention_rollout(attention_weights_list)
            
            # Extract attention to brain area tokens (exclude CLS token)
            brain_area_attention = cls_attention[:, 1:].cpu().numpy()  # (batch_size, n_brain_areas)
            
            # Add to collection
            remaining_samples = min(num_samples - samples_processed, batch_size)
            attention_matrices.append(brain_area_attention[:remaining_samples])
            animal_ids_list.extend(mouse_ids[:remaining_samples])
            samples_processed += remaining_samples
    
    # Concatenate all attention matrices
    all_attention = np.concatenate(attention_matrices, axis=0)
    all_animal_ids = np.array(animal_ids_list)
    
    logger.info(f"Extracted attention for {all_attention.shape[0]} samples")
    logger.info(f"Attention matrix shape: {all_attention.shape}")
    logger.info(f"Animal IDs shape: {all_animal_ids.shape}")
    logger.info(f"Unique animals: {len(np.unique(all_animal_ids))}")
    
    # Save if path provided
    if save_path is not None:
        # Save as dictionary with both attention and animal IDs
        save_dict = {
            'attention': all_attention,
            'animal_ids': all_animal_ids
        }
        np.save(save_path, save_dict, allow_pickle=True)
        logger.info(f"Saved attention matrix and animal IDs to {save_path}")
    
    return all_attention, all_animal_ids


def run_diagnosis(
    model: WidefieldTransformer,
    data_loader: DataLoader,
    device: torch.device,
    save_dir: str,
    num_classes: int = 2,
) -> Dict:
    """
    Run model diagnosis: confusion matrix, per-class metrics, per-animal accuracy.
    
    Args:
        model: Trained model
        data_loader: Data loader for evaluation (val or test)
        device: Device to run on
        save_dir: Directory to save diagnosis outputs
        num_classes: Number of classes (default 2)
        
    Returns:
        Diagnosis dictionary with metrics
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_logits = []
    all_mouse_ids = []
    
    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 2:
                data, labels = batch
                mouse_ids = ["unknown"] * data.size(0)
            elif len(batch) == 3:
                data, labels, mouse_ids = batch
            else:
                raise ValueError(f"Unexpected batch format: {len(batch)} elements")
            data = data.to(device)
            labels = labels.to(device)
            logits, _ = model(data)
            preds = logits.argmax(dim=-1)
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())
            all_mouse_ids.extend(mouse_ids)
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)
    all_mouse_ids = np.array(all_mouse_ids)
    
    # Confusion matrix
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, l in zip(all_predictions, all_labels):
        confusion[int(l), int(p)] += 1
    
    # Per-class metrics
    tp = np.sum((all_predictions == 1) & (all_labels == 1))
    fp = np.sum((all_predictions == 1) & (all_labels == 0))
    fn = np.sum((all_predictions == 0) & (all_labels == 1))
    tn = np.sum((all_predictions == 0) & (all_labels == 0))
    
    precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0.0
    
    precision_0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0.0
    macro_f1 = (f1_0 + f1_1) / 2
    accuracy = (all_predictions == all_labels).mean()
    
    # Per-animal accuracy (if mouse IDs available and not all "unknown")
    per_animal = {}
    if len(np.unique(all_mouse_ids)) > 1 or (len(np.unique(all_mouse_ids)) == 1 and "unknown" not in str(np.unique(all_mouse_ids)[0])):
        for mid in np.unique(all_mouse_ids):
            mask = all_mouse_ids == mid
            if mask.sum() > 0:
                acc = (all_predictions[mask] == all_labels[mask]).mean()
                per_animal[str(mid)] = {"accuracy": float(acc), "n_trials": int(mask.sum())}
    
    diagnosis = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "confusion_matrix": confusion.tolist(),
        "class_0": {"precision": precision_0, "recall": recall_0, "f1": f1_0},
        "class_1": {"precision": precision_1, "recall": recall_1, "f1": f1_1},
        "per_animal": per_animal,
    }
    
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "confusion_matrix.npy"), confusion)
    with open(os.path.join(save_dir, "diagnosis_report.json"), "w") as f:
        json.dump(diagnosis, f, indent=2)
    
    logger.info("Diagnosis complete:")
    logger.info(f"  Accuracy: {accuracy:.4f}, Macro F1: {macro_f1:.4f}")
    logger.info(f"  Confusion matrix:\n{confusion}")
    logger.info(f"  Saved to {save_dir}/diagnosis_report.json and confusion_matrix.npy")
    
    return diagnosis


def run_attention_and_diagnosis(
    model: WidefieldTransformer,
    data_loader: DataLoader,
    device: torch.device,
    save_dir: str,
    num_samples: int = 1000,
    num_classes: int = 2,
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """
    Extract attention rollout and run diagnosis on the selected model.
    
    Args:
        model: Trained model (best model recommended)
        data_loader: Data loader for evaluation
        device: Device to run on
        save_dir: Directory to save outputs
        num_samples: Number of samples for attention extraction
        num_classes: Number of classes for diagnosis
        
    Returns:
        Tuple of (diagnosis_dict, attention_matrix, animal_ids)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Attention extraction
    attention_path = os.path.join(save_dir, "attention_rollout.npy")
    attention_matrix, animal_ids = extract_attention_rollout(
        model, data_loader, device,
        num_samples=num_samples,
        save_path=attention_path,
    )
    logger.info(f"Attention rollout saved to {attention_path}")
    
    # Diagnosis
    diagnosis = run_diagnosis(
        model, data_loader, device,
        save_dir=save_dir,
        num_classes=num_classes,
    )
    
    return diagnosis, attention_matrix, animal_ids
