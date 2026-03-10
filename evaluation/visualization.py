"""
Visualization and analysis utilities for the widefield transformer model.
Ported from WidefieldModeling with adaptations for PRISMT.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

# Load atlas data (lazy loading)
_atlas = None
_atlas_allen = None
_complete_mask = None


def _load_atlas_data():
    """Lazy load atlas data."""
    global _atlas, _atlas_allen, _complete_mask
    if _atlas is None:
        atlas_path = Path(__file__).parent.parent / 'grid_values.npy'
        atlas_allen_path = Path(__file__).parent.parent / 'mask_atlas_new.npy'

        if not atlas_path.exists() or not atlas_allen_path.exists():
            logger.warning(f"Atlas files not found at {atlas_path} or {atlas_allen_path}")
            logger.warning("Brain map visualization will be skipped")
            return None, None, None

        _atlas = np.load(atlas_path, allow_pickle=True)
        _atlas_allen = np.load(atlas_allen_path, allow_pickle=True)
        _complete_mask = (_atlas_allen > 0).any(axis=2).astype(int)

    return _atlas, _atlas_allen, _complete_mask


def load_and_normalize_attention(attention_path: str) -> np.ndarray:
    """
    Load and normalize attention matrix.

    Args:
        attention_path: Path to .npy file containing attention matrix

    Returns:
        Normalized and thresholded attention array
    """
    data = np.load(attention_path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.dtype == object:
        # May be dict with 'attention' key
        if data.shape == () and isinstance(data.item(), dict):
            att = data.item().get('attention', data.item())
        else:
            att = data
    else:
        att = data
    if att.ndim == 0:
        att = np.atleast_2d(att)
    att = np.asarray(att, dtype=np.float64)
    att = att / att.max() if att.max() > 0 else att
    thresh = np.percentile(att, 95)
    mm = np.where(att > thresh, att, 0.0)
    return mm


_CDKL5_NAN_AREAS = frozenset([20, 21, 52, 53])


def _build_allen_hemi_pairs() -> List[Tuple[Optional[int], Optional[int]]]:
    """Build mapping from hemisphere-averaged area index to atlas L/R mask indices."""
    valid = [i for i in range(56) if i not in _CDKL5_NAN_AREAS]
    pairs: List[Tuple[Optional[int], Optional[int]]] = []
    for k in range(0, len(valid), 2):
        left = valid[k] if valid[k] < 52 else None
        right = valid[k + 1] if valid[k + 1] < 52 else None
        pairs.append((left, right))
    return pairs


def compute_brain_map(
    attention_data: np.ndarray,
    z_start: int,
    z_end: int,
    atlas_type: str = "grid"
) -> Optional[np.ndarray]:
    """
    Compute activation map for brain visualization.

    Args:
        attention_data: Attention data array (n_samples, n_brain_areas) or (n_brain_areas,)
        z_start: Start index for brain areas
        z_end: End index for brain areas
        atlas_type: 'grid' or 'allen'

    Returns:
        Combined brain map or None if atlas not available
    """
    atlas, atlas_allen, _ = _load_atlas_data()
    if atlas is None:
        return None

    if attention_data.ndim == 2:
        attention_data = attention_data.mean(axis=0)
    elif attention_data.ndim != 1:
        logger.warning(f"Unexpected attention data shape: {attention_data.shape}")
        return None

    if atlas_type == "allen":
        h, w = atlas_allen.shape[:2]
        n_areas = len(attention_data)
        n_atlas = atlas_allen.shape[2]

        owner = np.full((h, w), -1, dtype=int)
        owner_strength = np.zeros((h, w), dtype=float)

        def _update_owner(atlas_idx: int, area_weight: float) -> None:
            nonlocal parcel_weights
            mask = atlas_allen[:, :, atlas_idx]
            stronger = mask > owner_strength
            owner[stronger] = atlas_idx
            owner_strength[stronger] = mask[stronger]
            parcel_weights[atlas_idx] = area_weight

        parcel_weights: Dict[int, float] = {}

        if n_areas <= 26:
            hemi_pairs = _build_allen_hemi_pairs()
            for z in range(z_start, min(z_end, len(hemi_pairs))):
                weight = attention_data[z] if z < n_areas else 0.0
                left_idx, right_idx = hemi_pairs[z]
                if left_idx is not None and left_idx < n_atlas:
                    _update_owner(left_idx, weight)
                if right_idx is not None and right_idx < n_atlas:
                    _update_owner(right_idx, weight)
        else:
            for z in range(z_start, min(z_end, n_atlas)):
                weight = attention_data[z] if z < n_areas else 0.0
                _update_owner(z, weight)

        combined = np.zeros((h, w), dtype=float)
        for atlas_idx, weight in parcel_weights.items():
            combined[owner == atlas_idx] = weight
        return combined
    else:
        combined = np.zeros(atlas.shape[1:])
        for z in range(z_start, z_end):
            layer = z if z < 41 else (z - 41)
            if layer * 2 + 1 < atlas.shape[0]:
                weight = attention_data[z] if z < len(attention_data) else 0.0
                combined += weight * (atlas[layer * 2].T + atlas[layer * 2 + 1].T)
        return combined[30:-42, 24:-24]


def visualize_cls_attention(
    attention_matrix: np.ndarray,
    save_dir: Union[str, Path],
    task_name: str,
    phase1: str = "early",
    phase2: str = "late",
    stim_value: int = 1,
    keep_regions_mapping: Optional[List[int]] = None,
    atlas_type: str = "grid"
) -> Optional[Tuple[plt.Figure, Path]]:
    """
    Visualize CLS token attention as brain maps.

    Args:
        attention_matrix: Attention matrix of shape (n_samples, n_brain_areas).
        save_dir: Directory to save visualizations
        task_name: Task name for file naming
        phase1: First phase / class name
        phase2: Second phase / class name
        stim_value: Stimulus value
        keep_regions_mapping: Optional list mapping reduced indices to original brain area indices.
        atlas_type: 'grid' or 'allen'

    Returns:
        (figure, save_path) or None if atlas not available
    """
    atlas, atlas_allen, complete_mask = _load_atlas_data()
    if atlas is None:
        logger.warning("Atlas data not available, skipping brain map visualization")
        return None

    logger.info(f"Creating CLS token attention brain maps (atlas_type={atlas_type})...")

    cmap_mod1 = mcolors.LinearSegmentedColormap.from_list(
        'mod1_orange_auburn',
        [
            (0.00, '#ffffff'),
            (0.10, '#f3b199'),
            (0.20, '#f28b4c'),
            (0.40, '#e05c0d'),
            (0.60, '#a33900'),
            (1.00, '#000000'),
        ],
        N=256
    )

    att_normalized = attention_matrix / attention_matrix.max() if attention_matrix.max() > 0 else attention_matrix
    thresh = np.percentile(att_normalized, 95)
    att_thresholded = np.where(att_normalized > thresh, att_normalized, 0.0)
    att_mean = att_thresholded.mean(axis=0)

    if keep_regions_mapping is not None:
        n_original_areas = max(keep_regions_mapping) + 1 if keep_regions_mapping else len(att_mean)
        att_mean_mapped = np.zeros(n_original_areas)
        for reduced_idx, original_idx in enumerate(keep_regions_mapping):
            if reduced_idx < len(att_mean):
                att_mean_mapped[original_idx] = att_mean[reduced_idx]
        att_mean = att_mean_mapped
        logger.info(f"Mapped attention from {len(keep_regions_mapping)} reduced areas back to {n_original_areas} original areas")

    n_att_areas = len(att_mean)
    if atlas_type == "allen":
        n_areas = min(n_att_areas, atlas_allen.shape[2]) if n_att_areas > 26 else n_att_areas
    else:
        n_areas = 41

    brain_map = compute_brain_map(att_mean, 0, n_areas, atlas_type=atlas_type)

    if brain_map is None:
        logger.warning("Could not compute brain map")
        return None

    if atlas_type == "allen":
        brain_map = np.where(complete_mask > 0, brain_map, np.nan)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    cmap_copy = cmap_mod1.copy()
    cmap_copy.set_bad(color='white')
    im = ax.imshow(brain_map, cmap=cmap_copy, vmin=0, vmax=0.5, aspect='auto')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f'CLS Attention: {phase1} vs {phase2} (Stim={stim_value})', fontsize=14, fontweight='bold')

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Attention')
    plt.tight_layout()

    save_path_png = save_dir / f'attention_cls_{task_name}_stim{stim_value}.png'
    save_path_svg = save_dir / f'attention_cls_{task_name}_stim{stim_value}.svg'

    plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(save_path_svg, format='svg', dpi=300, bbox_inches='tight')
    logger.info(f"Saved CLS attention visualization to {save_path_png}")

    plt.close()
    return fig, save_path_png


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = False
) -> None:
    """
    Plot training and validation curves for loss, accuracy, and F1 score.

    Args:
        history: Dictionary containing training history
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    """
    has_f1_scores = 'val_f1_scores' in history and len(history['val_f1_scores']) > 0

    if has_f1_scores:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    epochs = range(1, len(history['train_losses']) + 1)

    ax1.plot(epochs, history['train_losses'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_losses'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history['train_accuracies'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_accuracies'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    val_accuracies = history['val_accuracies']
    best_val_acc = max(val_accuracies)
    if hasattr(val_accuracies, 'index'):
        best_epoch = val_accuracies.index(best_val_acc) + 1
    else:
        best_epoch = int(np.argmax(val_accuracies)) + 1
    ax2.annotate(
        f'Best: {best_val_acc:.3f} (Epoch {best_epoch})',
        xy=(best_epoch, best_val_acc),
        xytext=(best_epoch + len(epochs) * 0.1, best_val_acc + 0.05),
        arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7)
    )

    if has_f1_scores:
        ax3.plot(epochs, history['val_f1_scores'], 'g-', label='Validation F1 Score', linewidth=2)
        ax3.set_title('Validation F1 Score', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('F1 Score', fontsize=12)
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)

        val_f1_scores = history['val_f1_scores']
        best_val_f1 = max(val_f1_scores)
        if hasattr(val_f1_scores, 'index'):
            best_f1_epoch = val_f1_scores.index(best_val_f1) + 1
        else:
            best_f1_epoch = int(np.argmax(val_f1_scores)) + 1
        ax3.annotate(
            f'Best: {best_val_f1:.3f} (Epoch {best_f1_epoch})',
            xy=(best_f1_epoch, best_val_f1),
            xytext=(best_f1_epoch + len(epochs) * 0.1, best_val_f1 + 0.05),
            arrowprops=dict(arrowstyle='->', color='green', alpha=0.7),
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7)
        )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training curves to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_attention_heatmap(
    attention_matrix: np.ndarray,
    brain_area_names: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = False,
    title: str = "Attention Heatmap"
) -> None:
    """Plot attention weights as a heatmap."""
    mean_attention = np.mean(attention_matrix, axis=0)

    if brain_area_names is None:
        brain_area_names = [f"Area_{i + 1}" for i in range(len(mean_attention))]

    plt.figure(figsize=(12, 8))

    sns.heatmap(
        mean_attention.reshape(1, -1),
        xticklabels=brain_area_names,
        yticklabels=['CLS Token'],
        annot=True,
        fmt='.3f',
        cmap='viridis',
        cbar_kws={'label': 'Attention Weight'}
    )

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Brain Areas', fontsize=12)
    plt.ylabel('Source Token', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved attention heatmap to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_attention_distribution(
    attention_matrix: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = False
) -> None:
    """Plot distribution of attention weights across brain areas."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.boxplot(attention_matrix, labels=[f"A{i + 1}" for i in range(attention_matrix.shape[1])])
    ax1.set_title('Attention Distribution per Brain Area', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Brain Areas', fontsize=12)
    ax1.set_ylabel('Attention Weight', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)

    ax2.hist(attention_matrix.flatten(), bins=50, alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(attention_matrix), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(attention_matrix):.3f}')
    ax2.axvline(np.median(attention_matrix), color='green', linestyle='--', linewidth=2,
                label=f'Median: {np.median(attention_matrix):.3f}')
    ax2.set_title('Overall Attention Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Attention Weight', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved attention distribution plot to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = False
) -> None:
    """Plot confusion matrix as heatmap."""
    if class_names is None:
        n = confusion_matrix.shape[0]
        class_names = [f"Class {i}" for i in range(n)]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={'label': 'Count'}
    )
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix plot to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_per_animal_accuracy(
    per_animal: Dict[str, Dict],
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = False
) -> None:
    """Plot per-animal accuracy bar chart."""
    if not per_animal:
        logger.warning("No per-animal data to plot")
        return

    animals = list(per_animal.keys())
    accuracies = [per_animal[a]['accuracy'] for a in animals]
    n_trials = [per_animal[a]['n_trials'] for a in animals]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(animals))
    bars = ax.bar(x, accuracies, color='steelblue', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(animals, rotation=45, ha='right')
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xlabel('Animal', fontsize=12)
    ax.set_title('Per-Animal Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.axhline(y=np.mean(accuracies), color='red', linestyle='--', label=f'Mean: {np.mean(accuracies):.3f}')
    ax.legend()

    for i, (bar, n) in enumerate(zip(bars, n_trials)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'n={n}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved per-animal accuracy plot to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def analyze_attention_patterns(attention_matrix: np.ndarray) -> Dict:
    """Analyze attention patterns and return statistics."""
    n_samples, n_areas = attention_matrix.shape

    mean_attention = np.mean(attention_matrix, axis=0)
    std_attention = np.std(attention_matrix, axis=0)

    most_attended_idx = int(np.argmax(mean_attention))
    least_attended_idx = int(np.argmin(mean_attention))

    attention_entropy = []
    for i in range(n_samples):
        probs = attention_matrix[i] + 1e-10
        probs = probs / np.sum(probs)
        entropy = -np.sum(probs * np.log(probs))
        attention_entropy.append(entropy)

    attention_entropy = np.array(attention_entropy)

    attention_sparsity = []
    for i in range(n_samples):
        sorted_attention = np.sort(attention_matrix[i])
        n = len(sorted_attention)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_attention)) / (n * np.sum(sorted_attention)) - (n + 1) / n
        attention_sparsity.append(gini)

    attention_sparsity = np.array(attention_sparsity)

    mean_mean = np.mean(mean_attention)
    results = {
        'mean_attention': mean_attention,
        'std_attention': std_attention,
        'most_attended_area': most_attended_idx,
        'least_attended_area': least_attended_idx,
        'attention_entropy': attention_entropy,
        'attention_sparsity': attention_sparsity,
        'overall_stats': {
            'mean_entropy': float(np.mean(attention_entropy)),
            'mean_sparsity': float(np.mean(attention_sparsity)),
            'attention_concentration': float(np.max(mean_attention) / mean_mean) if mean_mean > 0 else 0.0
        }
    }

    logger.info("Attention Analysis Results:")
    logger.info(f"  Most attended area: {most_attended_idx} (attention: {mean_attention[most_attended_idx]:.3f})")
    logger.info(f"  Least attended area: {least_attended_idx} (attention: {mean_attention[least_attended_idx]:.3f})")

    return results


def create_comprehensive_report(
    attention_matrix: np.ndarray,
    save_dir: Union[str, Path],
    history: Optional[Dict[str, List[float]]] = None,
    model=None,
    diagnosis: Optional[Dict] = None,
    task_name: str = "task",
    phase1: str = "early",
    phase2: str = "late",
    stim_value: int = 1,
    atlas_type: str = "grid"
) -> None:
    """
    Create a comprehensive analysis report with all visualizations.

    Args:
        attention_matrix: Attention matrix from validation set
        save_dir: Directory to save results
        history: Optional training history (skipped if None, e.g. for HPO)
        model: Optional trained model (for architecture info)
        diagnosis: Optional diagnosis dict (for confusion matrix, per-animal)
        task_name: Task name for file naming
        phase1: First phase / class name
        phase2: Second phase / class name
        stim_value: Stimulus value
        atlas_type: 'grid' or 'allen'
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating comprehensive report in {save_dir}")

    # 1. Plot training curves (skip if no history, e.g. from HPO)
    if history is not None and len(history.get('train_losses', [])) > 0:
        plot_training_curves(
            history,
            save_path=str(save_dir / "training_curves.png"),
            show_plot=False
        )

    # 2. Plot attention heatmap
    plot_attention_heatmap(
        attention_matrix,
        save_path=str(save_dir / "attention_heatmap.png"),
        show_plot=False,
        title="CLS Token Attention to Brain Areas"
    )

    # 3. Plot attention distribution
    plot_attention_distribution(
        attention_matrix,
        save_path=str(save_dir / "attention_distribution.png"),
        show_plot=False
    )

    # 4. Analyze attention patterns
    attention_stats = analyze_attention_patterns(attention_matrix)

    # 5. Save attention statistics
    np.save(str(save_dir / "attention_statistics.npy"), attention_stats)

    # 6. CLS attention brain map
    visualize_cls_attention(
        attention_matrix,
        save_dir=save_dir,
        task_name=task_name,
        phase1=phase1,
        phase2=phase2,
        stim_value=stim_value,
        atlas_type=atlas_type
    )

    # 7. Confusion matrix plot
    if diagnosis is not None and 'confusion_matrix' in diagnosis:
        cm = np.array(diagnosis['confusion_matrix'])
        plot_confusion_matrix(
            cm,
            save_path=str(save_dir / "confusion_matrix.png"),
            show_plot=False
        )

    # 8. Per-animal accuracy plot
    if diagnosis is not None and diagnosis.get('per_animal'):
        plot_per_animal_accuracy(
            diagnosis['per_animal'],
            save_path=str(save_dir / "per_animal_accuracy.png"),
            show_plot=False
        )

    # 9. Create summary text report
    with open(save_dir / "analysis_report.txt", "w") as f:
        f.write("PRISMT Widefield Transformer Model Analysis Report\n")
        f.write("=" * 55 + "\n\n")

        if model is not None:
            f.write("Model Architecture:\n")
            f.write(f"  Brain Areas: {model.n_brain_areas}\n")
            f.write(f"  Time Points: {model.time_points}\n")
            f.write(f"  Hidden Dimension: {model.hidden_dim}\n")
            f.write(f"  Number of Heads: {model.num_heads}\n")
            f.write(f"  Number of Layers: {model.num_layers}\n")
            f.write(f"  Total Parameters: {sum(p.numel() for p in model.parameters()):,}\n\n")

        f.write("Training Results:\n")
        if history is not None and len(history.get('train_accuracies', [])) > 0:
            f.write(f"  Final Training Accuracy: {history['train_accuracies'][-1]:.4f}\n")
            f.write(f"  Best Validation Accuracy: {max(history['val_accuracies']):.4f}\n")
            f.write(f"  Total Epochs: {len(history['train_accuracies'])}\n\n")
        else:
            f.write("  (Training history not available, e.g. from HPO)\n\n")

        f.write("Attention Analysis:\n")
        f.write(f"  Most Attended Area: {attention_stats['most_attended_area']} ")
        f.write(f"(attention: {attention_stats['mean_attention'][attention_stats['most_attended_area']]:.3f})\n")
        f.write(f"  Least Attended Area: {attention_stats['least_attended_area']} ")
        f.write(f"(attention: {attention_stats['mean_attention'][attention_stats['least_attended_area']]:.3f})\n")
        f.write(f"  Mean Attention Entropy: {attention_stats['overall_stats']['mean_entropy']:.3f}\n")
        f.write(f"  Mean Attention Sparsity: {attention_stats['overall_stats']['mean_sparsity']:.3f}\n")
        f.write(f"  Attention Concentration: {attention_stats['overall_stats']['attention_concentration']:.3f}\n")

        if diagnosis is not None:
            f.write("\nDiagnosis:\n")
            f.write(f"  Accuracy: {diagnosis.get('accuracy', 0):.4f}\n")
            f.write(f"  Macro F1: {diagnosis.get('macro_f1', 0):.4f}\n")
            f.write(f"  Confusion Matrix:\n")
            for row in diagnosis.get('confusion_matrix', []):
                f.write(f"    {row}\n")

    logger.info("Comprehensive report created successfully!")
