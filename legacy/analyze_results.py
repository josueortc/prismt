#!/usr/bin/env python3
"""
PRISMT: Standalone analysis script for re-running post-training visualizations
on saved results.

Loads best_model.pt, attention_rollout.npy, diagnosis_report.json, and optionally
re-extracts attention with different --attention_samples when --data_path is provided.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from evaluation.visualization import create_comprehensive_report
from models.transformer import PRISMTransformer
from utils.helpers import get_device, setup_logging

logger = logging.getLogger(__name__)


def load_attention_matrix(results_dir: Path) -> Optional[np.ndarray]:
    """Load attention matrix from attention_rollout.npy."""
    path = results_dir / "attention_rollout.npy"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.dtype == object and data.shape == ():
        d = data.item()
        return d.get("attention", d) if isinstance(d, dict) else data
    return np.asarray(data, dtype=np.float64)


def load_diagnosis(results_dir: Path) -> Optional[Dict]:
    """Load diagnosis from diagnosis_report.json."""
    path = results_dir / "diagnosis_report.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _infer_model_config(state_dict: dict) -> Optional[Dict]:
    """Infer model config from state_dict (checkpoint does not store it explicitly)."""
    # token_embedding.projection: Linear(time_points, hidden_dim)
    key = "token_embedding.projection.weight"
    if key not in state_dict:
        return None
    time_points = state_dict[key].shape[1]
    hidden_dim = state_dict[key].shape[0]
    # positional_embedding.embeddings: (n_brain_areas + 1, hidden_dim)
    key = "positional_embedding.embeddings"
    if key not in state_dict:
        return None
    n_brain_areas = state_dict[key].shape[0] - 1
    # classifier: Linear(hidden_dim, num_classes)
    key = "classifier.weight"
    if key not in state_dict:
        return None
    num_classes = state_dict[key].shape[0]
    # num_heads from first transformer layer
    key = "transformer_layers.0.attention.query.weight"
    if key in state_dict:
        d = state_dict[key].shape[0]
        for nh in [2, 4, 8, 16]:
            if d % nh == 0:
                num_heads = nh
                break
        else:
            num_heads = 4
    else:
        num_heads = 4
    num_layers = 0
    for k in state_dict:
        if k.startswith("transformer_layers.") and ".attention.query.weight" in k:
            idx = int(k.split(".")[1])
            num_layers = max(num_layers, idx + 1)
    ff_dim = state_dict.get("transformer_layers.0.feed_forward.0.weight")
    ff_dim = ff_dim.shape[0] if ff_dim is not None else 256
    return {
        "n_brain_areas": n_brain_areas,
        "time_points": time_points,
        "hidden_dim": hidden_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "ff_dim": ff_dim,
        "num_classes": num_classes,
    }


def load_model_and_history(
    results_dir: Path, device: torch.device
) -> tuple[Optional[PRISMTransformer], Optional[Dict]]:
    """Load model and training history from best_model.pt."""
    path = results_dir / "best_model.pt"
    if not path.exists():
        return None, None
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "model_state_dict" not in ckpt:
        return None, None

    cfg = _infer_model_config(ckpt["model_state_dict"])
    if cfg is None:
        logger.warning("Could not infer model config from checkpoint")
        return None, None

    model = PRISMTransformer(
        n_brain_areas=cfg["n_brain_areas"],
        time_points=cfg["time_points"],
        hidden_dim=cfg["hidden_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        ff_dim=cfg["ff_dim"],
        num_classes=cfg["num_classes"],
        dropout=0.3,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    history = None
    if "train_losses" in ckpt and len(ckpt["train_losses"]) > 0:
        history = {
            "train_losses": ckpt["train_losses"],
            "train_accuracies": ckpt.get("train_accuracies", []),
            "val_losses": ckpt.get("val_losses", []),
            "val_accuracies": ckpt.get("val_accuracies", []),
            "val_f1_scores": ckpt.get("val_f1_scores", []),
        }

    return model, history


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-run post-training analysis on saved PRISMT results"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory containing best_model.pt, attention_rollout.npy, diagnosis_report.json",
    )
    parser.add_argument(
        "--atlas_type",
        type=str,
        default="grid",
        choices=["grid", "allen"],
        help="Atlas type for brain map visualization",
    )
    parser.add_argument(
        "--phase1",
        type=str,
        default="early",
        help="First phase name for file naming",
    )
    parser.add_argument(
        "--phase2",
        type=str,
        default="late",
        help="Second phase name for file naming",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="task",
        help="Task name for file naming",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use",
    )
    args = parser.parse_args()

    setup_logging()
    logger.info("PRISMT: Standalone analysis script")
    logger.info("=" * 60)

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logger.error("Results directory does not exist: %s", results_dir)
        sys.exit(1)

    device = get_device(args.device)

    # Load attention matrix
    attention_matrix = load_attention_matrix(results_dir)
    if attention_matrix is None:
        logger.error(
            "attention_rollout.npy not found in %s. "
            "Run training first or provide --data_path and --attention_samples to re-extract.",
            results_dir,
        )
        sys.exit(1)

    logger.info("Loaded attention matrix: shape %s", attention_matrix.shape)

    # Load diagnosis
    diagnosis = load_diagnosis(results_dir)
    if diagnosis is None:
        logger.warning("diagnosis_report.json not found; skipping confusion matrix and per-animal plots")

    # Load model and history
    model, history = load_model_and_history(results_dir, device)
    if model is None:
        logger.warning("best_model.pt not found or invalid; report will omit model architecture info")

    # Create comprehensive report
    create_comprehensive_report(
        attention_matrix=attention_matrix,
        save_dir=results_dir,
        history=history,
        model=model,
        diagnosis=diagnosis,
        task_name=args.task_name,
        phase1=args.phase1,
        phase2=args.phase2,
        stim_value=1,
        atlas_type=args.atlas_type,
    )

    logger.info("Analysis complete. Outputs saved to %s", results_dir)


if __name__ == "__main__":
    main()
