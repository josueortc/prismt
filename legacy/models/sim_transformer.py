"""
Backward-compatible alias. The model has been renamed from SIMTransformer to
PRISMTransformer (Pattern Reconstruction and Interpretation with a Structured
Multimodal Transformer). Import from prismt_transformer instead.
"""

from models.prismt_transformer import (  # noqa: F401
    ScalarTokenEmbedding,
    MaskedTokenEmbedding,
    PositionalEmbedding,
    CausalTemporalAttention,
    TransformerBlock,
    PRISMTransformer as SIMTransformer,
    create_prismt_model as create_sim_model,
)
