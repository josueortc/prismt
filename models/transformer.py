"""
Transformer model for widefield calcium imaging classification.
"""

import logging
import math
from typing import Tuple, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
        # Project each brain area's time series to hidden dimension
        # x: (batch_size, n_brain_areas, time_points) -> (batch_size, n_brain_areas, hidden_dim)
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
        
        # Add positional embeddings
        pos_embeddings = self.embeddings[:seq_len, :].unsqueeze(0)  # (1, seq_len, hidden_dim)
        x = x + pos_embeddings
        
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Initialize multi-head attention.
        
        Args:
            hidden_dim: Hidden dimension size
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply multi-head attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # Compute Q, K, V
        Q = self.query(x)  # (batch_size, seq_len, hidden_dim)
        K = self.key(x)    # (batch_size, seq_len, hidden_dim)
        V = self.value(x)  # (batch_size, seq_len, hidden_dim)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape: (batch_size, num_heads, seq_len, head_dim)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # Shape: (batch_size, num_heads, seq_len, seq_len)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        # Shape: (batch_size, num_heads, seq_len, head_dim)
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, hidden_dim
        )
        
        # Final linear transformation
        output = self.output(attended)
        
        # Average attention weights across heads for analysis
        attention_weights = attention_weights.mean(dim=1)  # (batch_size, seq_len, seq_len)
        
        return output, attention_weights


class TransformerBlock(nn.Module):
    """
    Single transformer decoder block.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        """
        Initialize transformer block.
        
        Args:
            hidden_dim: Hidden dimension size
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        # Self-attention with residual connection
        attn_out, attention_weights = self.attention(self.norm1(x), mask)
        x = x + attn_out
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(self.norm2(x))
        x = x + ff_out
        
        return x, attention_weights


class WidefieldTransformer(nn.Module):
    """
    Transformer model for widefield calcium imaging classification.
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
        dropout: float = 0.1
    ):
        """
        Initialize the transformer model.
        
        Args:
            n_brain_areas: Number of brain areas (after averaging)
            time_points: Number of time points per trial
            hidden_dim: Hidden dimension size
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            ff_dim: Feed-forward dimension
            num_classes: Number of output classes
            dropout: Dropout probability
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
        
        # Token embedding projects time series to hidden dimension
        self.token_embedding = TokenEmbedding(time_points, hidden_dim)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Positional embeddings (brain areas + CLS token)
        max_seq_len = n_brain_areas + 1
        self.positional_embedding = PositionalEmbedding(max_seq_len, hidden_dim)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self) -> None:
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    # Initialize classifier bias to small random values to reduce initial bias
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
        
        # Transpose to (batch_size, n_brain_areas, time_points) for tokenization
        x = x.transpose(1, 2)
        
        # Embed brain area time series as tokens
        tokens = self.token_embedding(x)  # (batch_size, n_brain_areas, hidden_dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, hidden_dim)
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # (batch_size, n_brain_areas + 1, hidden_dim)
        
        # Add positional embeddings
        tokens = self.positional_embedding(tokens)
        
        # Apply dropout
        tokens = self.dropout_layer(tokens)
        
        # Pass through transformer layers
        attention_weights_list = []
        for layer in self.transformer_layers:
            tokens, attention_weights = layer(tokens)
            attention_weights_list.append(attention_weights)
        
        # Extract CLS token for classification
        cls_output = tokens[:, 0, :]  # (batch_size, hidden_dim)
        
        # Apply classification head
        logits = self.classifier(cls_output)  # (batch_size, num_classes)
        
        return logits, attention_weights_list
    
    def get_attention_rollout(self, attention_weights_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute attention rollout to get final attention from CLS token to all other tokens.
        
        Args:
            attention_weights_list: List of attention weights from each layer
            
        Returns:
            Final attention weights from CLS token, shape (batch_size, seq_len)
        """
        # Start with identity matrix
        batch_size, seq_len, _ = attention_weights_list[0].shape
        rollout = torch.eye(seq_len).unsqueeze(0).expand(batch_size, -1, -1).to(attention_weights_list[0].device)
        
        # Multiply attention weights from all layers
        for attention_weights in attention_weights_list:
            # Add residual connection (0.5 * attention + 0.5 * identity)
            attention_with_residual = 0.5 * attention_weights + 0.5 * torch.eye(seq_len).unsqueeze(0).to(attention_weights.device)
            rollout = torch.matmul(attention_with_residual, rollout)
        
        # Extract attention from CLS token (first token) to all other tokens
        cls_attention = rollout[:, 0, :]  # (batch_size, seq_len)
        
        return cls_attention
    
    def get_model_info(self) -> str:
        """Get model information string."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = f"""
WidefieldTransformer Model Info:
- Brain Areas: {self.n_brain_areas}
- Time Points: {self.time_points}
- Hidden Dimension: {self.hidden_dim}
- Number of Heads: {self.num_heads}
- Number of Layers: {self.num_layers}
- Total Parameters: {total_params:,}
- Trainable Parameters: {trainable_params:,}
        """
        return info.strip()


def create_model(
    n_brain_areas: int,
    time_points: int,
    hidden_dim: int = 256,
    num_heads: int = 8,
    num_layers: int = 6,
    ff_dim: int = 1024,
    num_classes: int = 2,
    dropout: float = 0.1,
    device: torch.device = torch.device('cpu')
) -> WidefieldTransformer:
    """
    Create and initialize a WidefieldTransformer model.
    
    Args:
        n_brain_areas: Number of brain areas (after averaging)
        time_points: Number of time points per trial
        hidden_dim: Hidden dimension size
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        ff_dim: Feed-forward dimension
        num_classes: Number of output classes
        dropout: Dropout probability
        device: Device to place the model on
        
    Returns:
        Initialized WidefieldTransformer model
    """
    model = WidefieldTransformer(
        n_brain_areas=n_brain_areas,
        time_points=time_points,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=ff_dim,
        num_classes=num_classes,
        dropout=dropout
    )
    
    model = model.to(device)
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model
