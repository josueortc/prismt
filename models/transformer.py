"""
PRISMT (Pattern Reconstruction and Interpretation with a Structured Multimodal
Transformer) classification model for widefield calcium imaging.
Uses area-level tokenization with CausalTemporalAttention.
"""

import logging
from typing import Tuple, Optional, List
import torch
import torch.nn as nn

from models.prismt_transformer import CausalTemporalAttention, TransformerBlock

logger = logging.getLogger(__name__)


class TokenEmbedding(nn.Module):
    """
    Embedding layer that projects time series data to hidden dimensions.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Initialize token embedding.
        
        Args:
            input_dim: Number of time points per brain area
            hidden_dim: Hidden dimension size
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.projection = nn.Linear(input_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project brain area time series to hidden dimensions.
        
        Args:
            x: Input tensor of shape (batch_size, n_brain_areas, time_points)
            
        Returns:
            Embedded tokens of shape (batch_size, n_brain_areas, hidden_dim)
        """
        batch_size, n_areas, time_points = x.shape
        assert time_points == self.input_dim, f"Expected {self.input_dim} time points, got {time_points}"
        
        embedded = self.projection(x)
        
        return embedded


class PositionalEmbedding(nn.Module):
    """
    Learnable positional embeddings for transformer tokens.
    """
    
    def __init__(self, max_seq_len: int, hidden_dim: int):
        """
        Initialize positional embeddings.
        
        Args:
            max_seq_len: Maximum sequence length (including CLS token)
            hidden_dim: Hidden dimension size
        """
        super().__init__()
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Parameter(torch.randn(max_seq_len, hidden_dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional embeddings to input tokens.
        
        Args:
            x: Input tokens of shape (batch_size, seq_len, hidden_dim)
            
        Returns:
            Tokens with positional embeddings added
        """
        batch_size, seq_len, hidden_dim = x.shape
        assert seq_len <= self.max_seq_len, f"Sequence length {seq_len} exceeds max {self.max_seq_len}"
        assert hidden_dim == self.hidden_dim, f"Hidden dim mismatch: {hidden_dim} vs {self.hidden_dim}"
        
        pos_embeddings = self.embeddings[:seq_len, :].unsqueeze(0)
        x = x + pos_embeddings
        
        return x


class PRISMTransformer(nn.Module):
    """
    PRISMT (Pattern Reconstruction and Interpretation with a Structured Multimodal
    Transformer) classification model for widefield calcium imaging.
    Uses area-level tokenization with CausalTemporalAttention.
    Supports both classification and regression via task_mode.

    With area-level tokenization all tokens belong to a single effective
    timepoint, so the causal mask degenerates to full bidirectional attention --
    consistent mechanism, appropriate behaviour for classification.
    """
    
    def __init__(
        self,
        n_brain_areas: int,
        time_points: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        ff_dim: int = 1024,
        num_classes: int = 2,
        dropout: float = 0.1,
        task_mode: str = 'classification'
    ):
        """
        Initialize the PRISMTransformer classification model.
        
        Args:
            n_brain_areas: Number of brain areas (after averaging)
            time_points: Number of time points per trial
            hidden_dim: Hidden dimension size
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            ff_dim: Feed-forward dimension
            num_classes: Number of output classes (classification only)
            dropout: Dropout probability
            task_mode: 'classification' or 'regression'
        """
        super().__init__()
        
        self.n_brain_areas = n_brain_areas
        self.time_points = time_points
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.num_classes = num_classes
        self.dropout_prob = dropout
        self.task_mode = task_mode
        
        self.token_embedding = TokenEmbedding(time_points, hidden_dim)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        max_seq_len = n_brain_areas + 1
        self.positional_embedding = PositionalEmbedding(max_seq_len, hidden_dim)
        
        # Area-level tokenization: all tokens share a single effective timepoint
        # (time_points=1, n_brain_areas=seq_len including CLS) so the causal mask
        # allows full attention -- consistent with the scalar-token variant.
        seq_len = n_brain_areas + 1
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, ff_dim, time_points=1, n_brain_areas=seq_len, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        if task_mode == 'regression':
            self.classifier = nn.Linear(hidden_dim, 1)
        else:
            self.classifier = nn.Linear(hidden_dim, num_classes)
        
        self.dropout_layer = nn.Dropout(dropout)
        
        self._init_weights()
        
    def _init_weights(self) -> None:
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if module is self.classifier:
                        nn.init.uniform_(module.bias, -0.1, 0.1)
                    else:
                        nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through the transformer.
        
        Args:
            x: Input tensor of shape (batch_size, time_points, n_brain_areas)
            
        Returns:
            Tuple of (logits, attention_weights_list)
        """
        batch_size, time_points, n_brain_areas = x.shape
        assert time_points == self.time_points, f"Expected {self.time_points} time points, got {time_points}"
        assert n_brain_areas == self.n_brain_areas, f"Expected {self.n_brain_areas} brain areas, got {n_brain_areas}"
        
        x = x.transpose(1, 2)
        
        tokens = self.token_embedding(x)
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        
        tokens = self.positional_embedding(tokens)
        
        tokens = self.dropout_layer(tokens)
        
        attention_weights_list = []
        for layer in self.transformer_layers:
            tokens, attention_weights = layer(tokens, None)
            attention_weights_list.append(attention_weights)
        
        cls_output = tokens[:, 0, :]
        
        out = self.classifier(cls_output)
        if self.task_mode == 'regression':
            out = out.squeeze(-1)
        
        return out, attention_weights_list
    
    def get_attention_rollout(self, attention_weights_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute attention rollout to get final attention from CLS token to all other tokens.
        
        Args:
            attention_weights_list: List of attention weights from each layer
            
        Returns:
            Final attention weights from CLS token, shape (batch_size, seq_len)
        """
        batch_size, seq_len, _ = attention_weights_list[0].shape
        rollout = torch.eye(seq_len).unsqueeze(0).expand(batch_size, -1, -1).to(attention_weights_list[0].device)
        
        for attention_weights in attention_weights_list:
            attention_with_residual = 0.5 * attention_weights + 0.5 * torch.eye(seq_len).unsqueeze(0).to(attention_weights.device)
            rollout = torch.matmul(attention_with_residual, rollout)
        
        cls_attention = rollout[:, 0, :]
        
        return cls_attention
    
    def get_model_info(self) -> str:
        """Get model information string."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = f"""
PRISMTransformer Model Info:
- Brain Areas: {self.n_brain_areas}
- Time Points: {self.time_points}
- Hidden Dimension: {self.hidden_dim}
- Number of Heads: {self.num_heads}
- Number of Layers: {self.num_layers}
- Total Parameters: {total_params:,}
- Trainable Parameters: {trainable_params:,}
        """
        return info.strip()


# Backward-compatible aliases
WidefieldTransformer = PRISMTransformer


def create_prismt_model(
    n_brain_areas: int,
    time_points: int,
    hidden_dim: int = 256,
    num_heads: int = 8,
    num_layers: int = 6,
    ff_dim: int = 1024,
    num_classes: int = 2,
    dropout: float = 0.1,
    task_mode: str = 'classification',
    device: torch.device = torch.device('cpu')
) -> PRISMTransformer:
    """
    Create and initialize a PRISMTransformer classification model.
    
    Args:
        n_brain_areas: Number of brain areas (after averaging)
        time_points: Number of time points per trial
        hidden_dim: Hidden dimension size
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        ff_dim: Feed-forward dimension
        num_classes: Number of output classes (classification only)
        dropout: Dropout probability
        task_mode: 'classification' or 'regression'
        device: Device to place the model on
        
    Returns:
        Initialized PRISMTransformer model
    """
    model = PRISMTransformer(
        n_brain_areas=n_brain_areas,
        time_points=time_points,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=ff_dim,
        num_classes=num_classes,
        dropout=dropout,
        task_mode=task_mode
    )
    
    model = model.to(device)
    logger.info(f"Created PRISMTransformer model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model


# Backward-compatible alias
create_model = create_prismt_model
