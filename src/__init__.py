"""
Vision Transformer (ViT) - All Variants
A comprehensive TensorFlow/Keras implementation of Vision Transformers
"""

from .vit_all_variants import (
    VisionTransformer,
    PatchEmbedding,
    MultiHeadSelfAttention,
    MLP,
    TransformerBlock,
    create_vit,
    create_vit_s16,
    create_vit_s32,
    create_vit_b16,
    create_vit_b32,
    create_vit_l16,
    create_vit_l32,
    create_vit_h14,
    create_vit_h16,
    VIT_CONFIGS,
    list_variants,
    compare_variants,
)

__version__ = "1.0.0"
__author__ = "Vision Transformer Implementation"

__all__ = [
    "VisionTransformer",
    "PatchEmbedding",
    "MultiHeadSelfAttention",
    "MLP",
    "TransformerBlock",
    "create_vit",
    "create_vit_s16",
    "create_vit_s32",
    "create_vit_b16",
    "create_vit_b32",
    "create_vit_l16",
    "create_vit_l32",
    "create_vit_h14",
    "create_vit_h16",
    "VIT_CONFIGS",
    "list_variants",
    "compare_variants",
]
