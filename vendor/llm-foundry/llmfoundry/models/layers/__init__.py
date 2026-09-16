"""Dense layer registry used by the focused Gigar model."""

from llmfoundry.models.layers.attention import ATTN_CLASS_REGISTRY, RING_ATTN_CLASSES
from llmfoundry.models.layers.blocks import BLOCK_CLASS_REGISTRY
from llmfoundry.models.layers.custom_embedding import (
    EmbeddingParallelEmbedding,
    PARALLEL_EMBEDDING_REGISTRY,
    VocabParallelEmbedding,
)
from llmfoundry.models.layers.fc import (
    ColumnParallelLinear,
    FC_CLASS_REGISTRY,
    RowParallelLinear,
)
from llmfoundry.models.layers.mtp import MTP_CLASS_REGISTRY
from llmfoundry.models.layers.norm import LlamaRMSNorm

__all__ = [
    "ATTN_CLASS_REGISTRY",
    "BLOCK_CLASS_REGISTRY",
    "ColumnParallelLinear",
    "EmbeddingParallelEmbedding",
    "FC_CLASS_REGISTRY",
    "LlamaRMSNorm",
    "MTP_CLASS_REGISTRY",
    "PARALLEL_EMBEDDING_REGISTRY",
    "RING_ATTN_CLASSES",
    "RowParallelLinear",
    "VocabParallelEmbedding",
]
