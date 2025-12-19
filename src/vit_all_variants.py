"""
Vision Transformer (ViT) - All Common Variants
Includes Small, Base, Large, and Huge models with different patch sizes

Model Variants:
- ViT-S/16, ViT-S/32: Small models (~22M params)
- ViT-B/16, ViT-B/32: Base models (~86M params)
- ViT-L/16, ViT-L/32: Large models (~307M params)
- ViT-H/14, ViT-H/16: Huge models (~632M params)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class PatchEmbedding(layers.Layer):
    """Splits image into patches and embeds them."""
    def __init__(self, image_size=224, patch_size=16, embed_dim=768, **kwargs):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2

        self.projection = layers.Conv2D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            name="patch_projection"
        )

    def build(self, input_shape):
        """Build the patch embedding layer."""
        # Build the Conv2D projection layer
        self.projection.build(input_shape)
        super().build(input_shape)

    def call(self, x):
        x = self.projection(x)
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, [batch_size, -1, self.embed_dim])
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
        })
        return config


class MultiHeadSelfAttention(layers.Layer):
    """Multi-head self-attention mechanism."""
    def __init__(self, embed_dim, num_heads, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout,
            name="mha"
        )

    def build(self, input_shape):
        """Build the attention layer."""
        # Build the MultiHeadAttention layer (requires query_shape and value_shape)
        self.attention.build(query_shape=input_shape, value_shape=input_shape)
        super().build(input_shape)

    def call(self, x, training=False):
        return self.attention(x, x, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dropout": self.dropout_rate,
        })
        return config


class MLP(layers.Layer):
    """Multi-layer perceptron (Feed-forward network)."""
    def __init__(self, hidden_dim, embed_dim, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.dropout_rate = dropout

        self.dense1 = layers.Dense(hidden_dim, activation="gelu", name="dense1")
        self.dense2 = layers.Dense(embed_dim, name="dense2")
        self.dropout = layers.Dropout(dropout)

    def build(self, input_shape):
        """Build the MLP layers."""
        # Build dense layers
        self.dense1.build(input_shape)
        hidden_shape = input_shape[:-1] + (self.hidden_dim,)
        self.dense2.build(hidden_shape)
        super().build(input_shape)

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        x = self.dropout(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "embed_dim": self.embed_dim,
            "dropout": self.dropout_rate,
        })
        return config


class TransformerBlock(layers.Layer):
    """Transformer encoder block with attention and MLP."""
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout

        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.mlp = MLP(mlp_dim, embed_dim, dropout)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6, name="ln1")
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6, name="ln2")

    def build(self, input_shape):
        """Build the transformer block layers."""
        # Build layer normalization layers
        self.layernorm1.build(input_shape)
        self.layernorm2.build(input_shape)

        # Build attention and MLP
        self.attention.build(input_shape)
        self.mlp.build(input_shape)

        super().build(input_shape)

    def call(self, x, training=False):
        # Attention block with residual connection
        x_norm = self.layernorm1(x)
        attention_output = self.attention(x_norm, training=training)
        x = x + attention_output

        # MLP block with residual connection
        x_norm = self.layernorm2(x)
        mlp_output = self.mlp(x_norm, training=training)
        x = x + mlp_output

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim,
            "dropout": self.dropout_rate,
        })
        return config


class VisionTransformer(keras.Model):
    """
    Unified Vision Transformer implementation for all variants.

    Args:
        image_size: Input image size
        patch_size: Size of image patches
        num_layers: Number of transformer blocks
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        mlp_dim: Hidden dimension in MLP
        dropout: Dropout rate
        include_top: Whether to include classification head
        num_classes: Number of classes for classification
    """
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        num_layers=12,
        embed_dim=768,
        num_heads=12,
        mlp_dim=3072,
        dropout=0.1,
        include_top=False,
        num_classes=1000,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout
        self.include_top = include_top
        self.num_classes = num_classes

        self.num_patches = (image_size // patch_size) ** 2

        # Patch embedding
        self.patch_embedding = PatchEmbedding(image_size, patch_size, embed_dim)

        # Class token
        self.class_token = self.add_weight(
            shape=(1, 1, embed_dim),
            initializer="random_normal",
            trainable=True,
            name="class_token"
        )

        # Position embeddings
        self.position_embedding = self.add_weight(
            shape=(1, self.num_patches + 1, embed_dim),
            initializer="random_normal",
            trainable=True,
            name="position_embedding"
        )

        self.dropout = layers.Dropout(dropout)

        # Transformer blocks
        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout, name=f"transformer_block_{i}")
            for i in range(num_layers)
        ]

        # Final layer norm
        self.layernorm = layers.LayerNormalization(epsilon=1e-6, name="final_ln")

        # Classification head (optional)
        if include_top:
            self.head = layers.Dense(num_classes, name="classification_head")

    def build(self, input_shape):
        """Build the model layers."""
        # Build patch embedding layer
        self.patch_embedding.build(input_shape)

        # Build transformer blocks
        token_shape = (input_shape[0], self.num_patches + 1, self.embed_dim)
        for block in self.transformer_blocks:
            block.build(token_shape)

        # Build final layer norm
        self.layernorm.build(token_shape)

        # Build classification head if included
        if self.include_top:
            self.head.build((input_shape[0], self.embed_dim))

        # Mark the model as built
        super().build(input_shape)

    def call(self, x, training=False):
        batch_size = tf.shape(x)[0]

        # Create patches and embed
        x = self.patch_embedding(x)

        # Prepend class token
        class_tokens = tf.broadcast_to(self.class_token, [batch_size, 1, self.embed_dim])
        x = tf.concat([class_tokens, x], axis=1)

        # Add position embeddings
        x = x + self.position_embedding
        x = self.dropout(x, training=training)

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, training=training)

        # Final layer norm
        x = self.layernorm(x)

        # Extract class token or return all tokens
        if self.include_top:
            cls_token = x[:, 0]
            return self.head(cls_token)
        else:
            return x

    def extract_features(self, x, training=False):
        """Extract features from the model (class token representation)."""
        features = self.call(x, training=training)
        return features[:, 0]

    def get_config(self):
        config = super().get_config()
        config.update({
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "num_layers": self.num_layers,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim,
            "dropout": self.dropout_rate,
            "include_top": self.include_top,
            "num_classes": self.num_classes,
        })
        return config


# ============================================================================
# Model Configurations
# ============================================================================

VIT_CONFIGS = {
    # Small models (~22M parameters)
    "vit_s16": {
        "patch_size": 16,
        "num_layers": 12,
        "embed_dim": 384,
        "num_heads": 6,
        "mlp_dim": 1536,
    },
    "vit_s32": {
        "patch_size": 32,
        "num_layers": 12,
        "embed_dim": 384,
        "num_heads": 6,
        "mlp_dim": 1536,
    },

    # Base models (~86M parameters)
    "vit_b16": {
        "patch_size": 16,
        "num_layers": 12,
        "embed_dim": 768,
        "num_heads": 12,
        "mlp_dim": 3072,
    },
    "vit_b32": {
        "patch_size": 32,
        "num_layers": 12,
        "embed_dim": 768,
        "num_heads": 12,
        "mlp_dim": 3072,
    },

    # Large models (~307M parameters)
    "vit_l16": {
        "patch_size": 16,
        "num_layers": 24,
        "embed_dim": 1024,
        "num_heads": 16,
        "mlp_dim": 4096,
    },
    "vit_l32": {
        "patch_size": 32,
        "num_layers": 24,
        "embed_dim": 1024,
        "num_heads": 16,
        "mlp_dim": 4096,
    },

    # Huge models (~632M parameters)
    "vit_h14": {
        "patch_size": 14,
        "num_layers": 32,
        "embed_dim": 1280,
        "num_heads": 16,
        "mlp_dim": 5120,
    },
    "vit_h16": {
        "patch_size": 16,
        "num_layers": 32,
        "embed_dim": 1280,
        "num_heads": 16,
        "mlp_dim": 5120,
    },
}


# ============================================================================
# Factory Functions for All Variants
# ============================================================================

def create_vit(
    variant="vit_b16",
    image_size=224,
    include_top=False,
    num_classes=1000,
    dropout=0.1,
    weights=None
):
    """
    Create any Vision Transformer variant.

    Args:
        variant: Model variant name (e.g., 'vit_b16', 'vit_l32')
        image_size: Input image size
        include_top: Whether to include classification head
        num_classes: Number of classes (if include_top=True)
        dropout: Dropout rate
        weights: Path to pre-trained weights (optional)

    Returns:
        VisionTransformer model
    """
    if variant not in VIT_CONFIGS:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(VIT_CONFIGS.keys())}")

    config = VIT_CONFIGS[variant]

    model = VisionTransformer(
        image_size=image_size,
        patch_size=config["patch_size"],
        num_layers=config["num_layers"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        mlp_dim=config["mlp_dim"],
        dropout=dropout,
        include_top=include_top,
        num_classes=num_classes
    )

    # Build the model
    model.build((None, image_size, image_size, 3))

    # Load weights if provided
    if weights is not None:
        model.load_weights(weights)

    return model


# Specific factory functions for each variant

def create_vit_s16(image_size=224, include_top=False, num_classes=1000, dropout=0.1, weights=None):
    """ViT-Small/16: Small model with 16x16 patches (~22M params)"""
    return create_vit("vit_s16", image_size, include_top, num_classes, dropout, weights)


def create_vit_s32(image_size=224, include_top=False, num_classes=1000, dropout=0.1, weights=None):
    """ViT-Small/32: Small model with 32x32 patches (~22M params)"""
    return create_vit("vit_s32", image_size, include_top, num_classes, dropout, weights)


def create_vit_b16(image_size=224, include_top=False, num_classes=1000, dropout=0.1, weights=None):
    """ViT-Base/16: Base model with 16x16 patches (~86M params)"""
    return create_vit("vit_b16", image_size, include_top, num_classes, dropout, weights)


def create_vit_b32(image_size=224, include_top=False, num_classes=1000, dropout=0.1, weights=None):
    """ViT-Base/32: Base model with 32x32 patches (~86M params)"""
    return create_vit("vit_b32", image_size, include_top, num_classes, dropout, weights)


def create_vit_l16(image_size=224, include_top=False, num_classes=1000, dropout=0.1, weights=None):
    """ViT-Large/16: Large model with 16x16 patches (~307M params)"""
    return create_vit("vit_l16", image_size, include_top, num_classes, dropout, weights)


def create_vit_l32(image_size=224, include_top=False, num_classes=1000, dropout=0.1, weights=None):
    """ViT-Large/32: Large model with 32x32 patches (~307M params)"""
    return create_vit("vit_l32", image_size, include_top, num_classes, dropout, weights)


def create_vit_h14(image_size=224, include_top=False, num_classes=1000, dropout=0.1, weights=None):
    """ViT-Huge/14: Huge model with 14x14 patches (~632M params)"""
    return create_vit("vit_h14", image_size, include_top, num_classes, dropout, weights)


def create_vit_h16(image_size=224, include_top=False, num_classes=1000, dropout=0.1, weights=None):
    """ViT-Huge/16: Huge model with 16x16 patches (~632M params)"""
    return create_vit("vit_h16", image_size, include_top, num_classes, dropout, weights)


# ============================================================================
# Utility Functions
# ============================================================================

def list_variants():
    """List all available ViT variants with their specifications."""
    print("\n" + "="*80)
    print("Available Vision Transformer Variants")
    print("="*80 + "\n")

    for variant_name, config in VIT_CONFIGS.items():
        model = create_vit(variant_name, image_size=224, include_top=False)
        params = model.count_params()

        print(f"{variant_name.upper()}")
        print(f"  Patch size: {config['patch_size']}x{config['patch_size']}")
        print(f"  Layers: {config['num_layers']}")
        print(f"  Embed dim: {config['embed_dim']}")
        print(f"  Attention heads: {config['num_heads']}")
        print(f"  MLP dim: {config['mlp_dim']}")
        print(f"  Parameters: {params:,}")
        print()


def compare_variants(image_size=224):
    """Compare all variants side by side."""
    print("\n" + "="*100)
    print(f"{'Variant':<12} {'Patch':<8} {'Layers':<8} {'Embed':<8} {'Heads':<8} {'Params':<15} {'Tokens':<10}")
    print("="*100)

    for variant_name in VIT_CONFIGS.keys():
        config = VIT_CONFIGS[variant_name]
        model = create_vit(variant_name, image_size=image_size, include_top=False)
        params = model.count_params()
        num_patches = (image_size // config['patch_size']) ** 2
        num_tokens = num_patches + 1  # +1 for class token

        print(f"{variant_name.upper():<12} "
              f"{config['patch_size']:<8} "
              f"{config['num_layers']:<8} "
              f"{config['embed_dim']:<8} "
              f"{config['num_heads']:<8} "
              f"{params:>12,}   "
              f"{num_tokens:<10}")

    print("="*100 + "\n")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Vision Transformer - All Variants")
    print("="*80)

    # List all variants
    list_variants()

    # Compare variants
    compare_variants(image_size=224)

    # Test a few models
    print("\n" + "="*80)
    print("Testing Models")
    print("="*80 + "\n")

    test_variants = ["vit_s16", "vit_b32", "vit_l16"]
    dummy_input = tf.random.normal((1, 224, 224, 3))

    for variant in test_variants:
        print(f"\nTesting {variant.upper()}...")
        model = create_vit(variant, image_size=224, include_top=False)
        output = model(dummy_input, training=False)
        print(f"  Input: {dummy_input.shape}")
        print(f"  Output: {output.shape}")
        print(f"  Parameters: {model.count_params():,}")
