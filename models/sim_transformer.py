"""
SIM (Simple Image Masking) Transformer model for widefield calcium imaging.
Each timepoint becomes a token, with masking and causal temporal attention.
"""

import logging
import math
from typing import Tuple, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ScalarTokenEmbedding(nn.Module):
    """
    Embedding layer that projects each scalar (brain area × timepoint) to hidden dimensions.
    Each scalar becomes a single token.
    For 10 brain areas and 20 timepoints, this creates 200 tokens.
    """
    
    def __init__(self, hidden_dim: int):
        """
        Initialize scalar token embedding.
        
        Args:
            hidden_dim: Hidden dimension size
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        # Project each scalar (1D value) to hidden dimension
        self.projection = nn.Linear(1, hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project scalar values to hidden dimensions.
        
        Args:
            x: Input tensor of shape (batch_size, time_points, n_brain_areas)
            
        Returns:
            Embedded tokens of shape (batch_size, time_points * n_brain_areas, hidden_dim)
        """
        batch_size, time_points, n_brain_areas = x.shape
        
        # Flatten to (batch_size, time_points * n_brain_areas) - each element is a scalar
        x_flat = x.reshape(batch_size, -1)  # (batch_size, time_points * n_brain_areas)
        
        # Add dimension for linear projection: (batch_size, time_points * n_brain_areas, 1)
        x_flat = x_flat.unsqueeze(-1)
        
        # Project each scalar to hidden dimension
        # (batch_size, time_points * n_brain_areas, 1) -> (batch_size, time_points * n_brain_areas, hidden_dim)
        embedded = self.projection(x_flat)
        
        return embedded


class MaskedTokenEmbedding(nn.Module):
    """
    Learnable masked token embedding for reconstruction task.
    
    This is a SINGLE learnable token that substitutes ALL masked token positions.
    Following SimMIM: mask_token is a 1D parameter (hidden_dim,) that gets expanded
    to all positions. Positional embeddings are added separately to differentiate positions.
    
    Key property: There is only ONE learnable masked token parameter shared across
    all masked positions. This ensures consistent masking behavior.
    """
    
    def __init__(self, hidden_dim: int):
        """
        Initialize masked token embedding.
        
        Args:
            hidden_dim: Hidden dimension size
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        # SINGLE learnable masked token parameter shared across all masked positions
        # Shape: (hidden_dim,) - this is the only learnable parameter for masking
        self.masked_token = nn.Parameter(torch.randn(hidden_dim))
        
    def forward(self, batch_size: int, num_patches: int) -> torch.Tensor:
        """
        Get masked token embeddings (without positional embeddings).
        
        Expands the single masked token to all positions. All masked positions will
        use the same base masked token (positional embeddings are added separately).
        
        Args:
            batch_size: Batch size
            num_patches: Number of patches/tokens
            
        Returns:
            Masked token embeddings of shape (batch_size, num_patches, hidden_dim)
            All positions contain the same base masked token (before positional embeddings)
        """
        # Expand single masked token to all positions: 'd -> b n d'
        # Using expand() creates views (not copies), so all positions share the same base token
        mask_tokens = self.masked_token.unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_dim)
        mask_tokens = mask_tokens.expand(batch_size, num_patches, -1)  # (batch_size, num_patches, hidden_dim)
        return mask_tokens


class PositionalEmbedding(nn.Module):
    """
    Learnable positional embeddings for transformer tokens.
    Each token has a position defined by (timepoint, brain_area).
    We use separate embeddings for timepoint and brain area, then add them.
    """
    
    def __init__(self, time_points: int, n_brain_areas: int, hidden_dim: int):
        """
        Initialize positional embeddings.
        
        Args:
            time_points: Number of timepoints
            n_brain_areas: Number of brain areas
            hidden_dim: Hidden dimension size
        """
        super().__init__()
        self.time_points = time_points
        self.n_brain_areas = n_brain_areas
        self.hidden_dim = hidden_dim
        # Separate embeddings for timepoint and brain area positions
        self.timepoint_embedding = nn.Parameter(torch.randn(1, time_points, hidden_dim))
        self.brain_area_embedding = nn.Parameter(torch.randn(1, n_brain_areas, hidden_dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional embeddings to input tokens.
        Tokens are arranged as: [t0_b0, t0_b1, ..., t0_bN, t1_b0, t1_b1, ..., tT_bN]
        where t = timepoint, b = brain area.
        
        Args:
            x: Input tensor of shape (batch_size, time_points * n_brain_areas, hidden_dim)
            
        Returns:
            Output tensor of shape (batch_size, time_points * n_brain_areas, hidden_dim)
        """
        batch_size, seq_len, hidden_dim = x.shape
        total_tokens = self.time_points * self.n_brain_areas
        assert seq_len == total_tokens, f"Sequence length {seq_len} != expected {total_tokens}"
        assert hidden_dim == self.hidden_dim, f"Hidden dim mismatch: {hidden_dim} vs {self.hidden_dim}"
        
        # Reshape to (batch_size, time_points, n_brain_areas, hidden_dim)
        x_reshaped = x.reshape(batch_size, self.time_points, self.n_brain_areas, hidden_dim)
        
        # Add timepoint embeddings: (batch_size, time_points, n_brain_areas, hidden_dim)
        timepoint_emb = self.timepoint_embedding.unsqueeze(2)  # (1, time_points, 1, hidden_dim)
        x_reshaped = x_reshaped + timepoint_emb
        
        # Add brain area embeddings: (batch_size, time_points, n_brain_areas, hidden_dim)
        brain_area_emb = self.brain_area_embedding.unsqueeze(1)  # (1, 1, n_brain_areas, hidden_dim)
        x_reshaped = x_reshaped + brain_area_emb
        
        # Reshape back to (batch_size, time_points * n_brain_areas, hidden_dim)
        x = x_reshaped.reshape(batch_size, seq_len, hidden_dim)
        
        return x


class CausalTemporalAttention(nn.Module):
    """
    Multi-head self-attention with causal temporal masking.
    Following CausalAttention pattern: tokens can attend to themselves and all previous timepoints.
    Uses multiplicative masking after softmax (like the reference implementation).
    """
    
    def __init__(self, hidden_dim: int, num_heads: int, time_points: int, n_brain_areas: int, dropout: float = 0.1):
        """
        Initialize causal temporal attention.
        
        Args:
            hidden_dim: Hidden dimension size
            num_heads: Number of attention heads
            time_points: Number of timepoints
            n_brain_areas: Number of brain areas
            dropout: Dropout probability
        """
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.time_points = time_points
        self.n_brain_areas = n_brain_areas
        
        # Following CausalAttention: use single linear layer for QKV
        inner_dim = self.head_dim * num_heads
        self.to_qkv = nn.Linear(hidden_dim, inner_dim * 3, bias=False)
        
        # Output projection
        project_out = not (num_heads == 1 and self.head_dim == hidden_dim)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, hidden_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        
        # Scale factor: following CausalAttention pattern (dim_head ** -0.5)
        self.scale = self.head_dim ** -0.5
        
        # Create causal mask: tokens from same timepoint can attend to each other,
        # tokens from timepoint t can attend to tokens from timepoint s where s <= t
        self.register_buffer('causal_mask', self._create_causal_mask(time_points, n_brain_areas))
    
    def _create_causal_mask(self, time_points: int, n_brain_areas: int) -> torch.Tensor:
        """
        Create causal temporal attention mask.
        Tokens from the same timepoint can attend to each other (all-to-all).
        Tokens from timepoint t can attend to tokens from timepoint s where s <= t.
        
        Token order: [t0_b0, t0_b1, ..., t0_bN, t1_b0, t1_b1, ..., tT_bN]
        
        Args:
            time_points: Number of timepoints
            n_brain_areas: Number of brain areas
            
        Returns:
            Causal mask of shape (time_points * n_brain_areas, time_points * n_brain_areas)
            where 1 = can attend, 0 = cannot
        """
        total_tokens = time_points * n_brain_areas
        mask = torch.zeros(total_tokens, total_tokens)
        
        for t_i in range(time_points):
            for t_j in range(time_points):
                # Tokens from timepoint t_i can attend to tokens from timepoint t_j if t_j <= t_i
                if t_j <= t_i:
                    # All brain areas at t_i can attend to all brain areas at t_j
                    start_i = t_i * n_brain_areas
                    end_i = (t_i + 1) * n_brain_areas
                    start_j = t_j * n_brain_areas
                    end_j = (t_j + 1) * n_brain_areas
                    mask[start_i:end_i, start_j:end_j] = 1.0
        
        return mask
    
    def forward(
        self, 
        x: torch.Tensor, 
        timepoint_indices: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply causal temporal attention.
        Following CausalAttention pattern: multiplicative masking after softmax.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            timepoint_indices: Optional tensor (kept for API compatibility, not used)
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # Compute QKV in one go (following CausalAttention pattern)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = qkv
        
        # Reshape for multi-head attention: 'b n (h d) -> b h n d'
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape: (batch_size, num_heads, seq_len, head_dim)
        
        # Compute attention scores
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # Shape: (batch_size, num_heads, seq_len, seq_len)
        
        # Apply softmax first (following CausalAttention pattern)
        attn = self.attend(dots)
        attn = self.dropout(attn)
        
        # Apply causal mask multiplicatively AFTER softmax (following CausalAttention pattern)
        # Reshape mask: 'w1 h1 -> 1 1 w1 h1'
        mask = self.causal_mask[:seq_len, :seq_len].unsqueeze(0).unsqueeze(0)
        # Ensure mask is on the same device as attn
        mask = mask.to(attn.device)
        attn = attn * mask
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        # Shape: (batch_size, num_heads, seq_len, head_dim)
        
        # Reshape: 'b h n d -> b n (h d)'
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.head_dim * self.num_heads)
        
        # Final output projection
        output = self.to_out(out)
        
        # Average attention weights across heads for analysis
        attention_weights = attn.mean(dim=1)  # (batch_size, seq_len, seq_len)
        
        return output, attention_weights


class TransformerBlock(nn.Module):
    """
    Single transformer block with causal temporal attention.
    Following PreNorm pattern from reference implementation.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int, time_points: int, n_brain_areas: int, dropout: float = 0.1):
        """
        Initialize transformer block.
        
        Args:
            hidden_dim: Hidden dimension size
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            time_points: Number of timepoints
            n_brain_areas: Number of brain areas
            dropout: Dropout probability
        """
        super().__init__()
        
        self.attention = CausalTemporalAttention(hidden_dim, num_heads, time_points, n_brain_areas, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(
        self, 
        x: torch.Tensor, 
        timepoint_indices: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through transformer block.
        Following PreNorm pattern: normalize before applying function.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            timepoint_indices: Optional tensor (kept for API compatibility, not used)
            
        Returns:
            Tuple of (output, attention_weights)
        """
        # Self-attention with residual connection (PreNorm pattern)
        attn_out, attention_weights = self.attention(self.norm1(x), timepoint_indices)
        x = x + attn_out
        
        # Feed-forward with residual connection (PreNorm pattern)
        ff_out = self.feed_forward(self.norm2(x))
        x = x + ff_out
        
        return x, attention_weights


class SIMTransformer(nn.Module):
    """
    SIM Transformer model for widefield calcium imaging reconstruction.
    Each scalar (brain area × timepoint) becomes a token.
    For 10 brain areas and 20 timepoints, this creates 200 tokens.
    """
    
    def __init__(
        self,
        n_brain_areas: int,
        time_points: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        ff_dim: int = 1024,
        dropout: float = 0.1
    ):
        """
        Initialize the SIM transformer model.
        
        Args:
            n_brain_areas: Number of brain areas
            time_points: Number of time points per trial
            hidden_dim: Hidden dimension size
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            ff_dim: Feed-forward dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        self.n_brain_areas = n_brain_areas
        self.time_points = time_points
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.dropout_prob = dropout
        self.total_tokens = time_points * n_brain_areas
        
        # Token embedding: each scalar becomes a token
        self.token_embedding = ScalarTokenEmbedding(hidden_dim)
        
        # Masked token embedding
        self.masked_token_embedding = MaskedTokenEmbedding(hidden_dim)
        
        # Positional embeddings (encode both timepoint and brain area positions)
        self.positional_embedding = PositionalEmbedding(time_points, n_brain_areas, hidden_dim)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, ff_dim, time_points, n_brain_areas, dropout)
            for _ in range(num_layers)
        ])
        
        # Reconstruction head: predict scalar value for each token
        self.reconstruction_head = nn.Linear(hidden_dim, 1)
        
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
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through the transformer.
        
        Tokenization:
        - Each scalar (brain area × timepoint) becomes a token
        - For 10 brain areas and 20 timepoints, this creates 200 tokens
        - Token order: [t0_b0, t0_b1, ..., t0_bN, t1_b0, t1_b1, ..., tT_bN]
        
        Masking logic:
        - Masks are specified at brain-area level: (batch_size, time_points, n_brain_areas)
        - Masked values are zeroed out BEFORE tokenization
        - Masked tokens are replaced with learnable masked token embeddings
        
        Causal attention:
        - Tokens from same timepoint can attend to each other (all-to-all)
        - Tokens from timepoint t can attend to tokens from timepoint s where s <= t
        
        Args:
            x: Input tensor of shape (batch_size, time_points, n_brain_areas)
            mask: Optional mask tensor of shape (batch_size, time_points, n_brain_areas) 
                  where 1 = masked, 0 = unmasked. If shape is (batch_size, time_points),
                  it will be expanded to mask all brain areas at masked timepoints.
            
        Returns:
            Tuple of (reconstructed_output, attention_weights_list)
            reconstructed_output: (batch_size, time_points, n_brain_areas)
        """
        batch_size, time_points, n_brain_areas = x.shape
        assert time_points == self.time_points, f"Expected {self.time_points} time points, got {time_points}"
        assert n_brain_areas == self.n_brain_areas, f"Expected {self.n_brain_areas} brain areas, got {n_brain_areas}"
        
        # Handle mask: apply brain-area-level masking BEFORE tokenization
        if mask is not None:
            # If mask is 2D (batch_size, time_points), expand to 3D to mask all brain areas
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1).expand(-1, -1, n_brain_areas)  # (batch_size, time_points, n_brain_areas)
            
            assert mask.shape == x.shape, f"Mask shape {mask.shape} must match input shape {x.shape}"
            
            # Apply masking to input data: replace masked brain areas with zeros
            masked_x = x * (1.0 - mask)  # (batch_size, time_points, n_brain_areas)
            
            # Flatten mask to match token order: (batch_size, time_points * n_brain_areas)
            token_mask = mask.reshape(batch_size, -1)  # (batch_size, time_points * n_brain_areas)
        else:
            masked_x = x
            token_mask = None
        
        # Embed scalars as tokens: each scalar becomes a token
        tokens = self.token_embedding(masked_x)  # (batch_size, time_points * n_brain_areas, hidden_dim)
        
        # Add positional embeddings (encodes both timepoint and brain area positions)
        tokens = self.positional_embedding(tokens)  # (batch_size, time_points * n_brain_areas, hidden_dim)
        
        # Apply token-level masking: replace masked tokens with learnable masked token
        if token_mask is not None:
            # Create mask tokens: SINGLE masked token expanded to all positions
            mask_tokens = self.masked_token_embedding(batch_size, self.total_tokens)  # (batch_size, total_tokens, hidden_dim)
            # Add positional embeddings to mask tokens
            mask_tokens = self.positional_embedding(mask_tokens)  # (batch_size, total_tokens, hidden_dim)
            
            # Replace masked tokens using torch.where
            # token_mask: (batch_size, total_tokens) -> expand to (batch_size, total_tokens, hidden_dim)
            token_mask_expanded = token_mask.unsqueeze(-1).expand(-1, -1, self.hidden_dim)  # (batch_size, total_tokens, hidden_dim)
            tokens = torch.where(token_mask_expanded.bool(), mask_tokens, tokens)
        
        # Apply dropout
        tokens = self.dropout_layer(tokens)
        
        # Pass through transformer layers
        # Note: timepoint_indices not needed anymore since causal mask is pre-computed
        attention_weights_list = []
        for layer in self.transformer_layers:
            tokens, attention_weights = layer(tokens, None)
            attention_weights_list.append(attention_weights)
        
        # Reconstruct scalar values for each token
        reconstructed_flat = self.reconstruction_head(tokens)  # (batch_size, time_points * n_brain_areas, 1)
        reconstructed_flat = reconstructed_flat.squeeze(-1)  # (batch_size, time_points * n_brain_areas)
        
        # Reshape back to (batch_size, time_points, n_brain_areas)
        reconstructed = reconstructed_flat.reshape(batch_size, time_points, n_brain_areas)
        
        return reconstructed, attention_weights_list
    
    def get_model_info(self) -> str:
        """Get model information string."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = f"""
SIMTransformer Model Info:
- Brain Areas: {self.n_brain_areas}
- Time Points: {self.time_points}
- Hidden Dimension: {self.hidden_dim}
- Number of Heads: {self.num_heads}
- Number of Layers: {self.num_layers}
- Total Parameters: {total_params:,}
- Trainable Parameters: {trainable_params:,}
        """
        return info.strip()


def create_sim_model(
    n_brain_areas: int,
    time_points: int,
    hidden_dim: int = 256,
    num_heads: int = 8,
    num_layers: int = 6,
    ff_dim: int = 1024,
    dropout: float = 0.1,
    device: torch.device = torch.device('cpu')
) -> SIMTransformer:
    """
    Create and initialize a SIMTransformer model.
    
    Args:
        n_brain_areas: Number of brain areas
        time_points: Number of time points per trial
        hidden_dim: Hidden dimension size
        num_heads: Number of attention heads (must divide hidden_dim)
        num_layers: Number of transformer layers
        ff_dim: Feed-forward dimension
        dropout: Dropout probability
        device: Device to place the model on
        
    Returns:
        Initialized SIMTransformer model
    """
    # Validate and adjust num_heads if needed
    if hidden_dim % num_heads != 0:
        # Find the largest divisor of hidden_dim that is <= num_heads
        adjusted_num_heads = num_heads
        while hidden_dim % adjusted_num_heads != 0 and adjusted_num_heads > 1:
            adjusted_num_heads -= 1
        
        # If we couldn't find a divisor <= num_heads, find the largest divisor of hidden_dim
        if adjusted_num_heads == 1:
            # Find all divisors and pick the largest one <= hidden_dim
            adjusted_num_heads = 1
            for i in range(1, hidden_dim + 1):
                if hidden_dim % i == 0 and i <= num_heads:
                    adjusted_num_heads = max(adjusted_num_heads, i)
        
        logger.warning(
            f"num_heads={num_heads} is not divisible by hidden_dim={hidden_dim}. "
            f"Adjusting num_heads to {adjusted_num_heads}."
        )
        num_heads = adjusted_num_heads
    
    model = SIMTransformer(
        n_brain_areas=n_brain_areas,
        time_points=time_points,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=ff_dim,
        dropout=dropout
    )
    
    model = model.to(device)
    logger.info(f"Created SIM model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model

