# Vision Transformer (ViT) - All Variants

A comprehensive TensorFlow/Keras implementation of Vision Transformer models including Small, Base, Large, and Huge variants with different patch sizes.

## 🎯 Overview

This repository provides a clean, modular implementation of Vision Transformers (ViT) that supports all standard variants from the original paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale".

### Available Models

| Model | Patch Size | Layers | Embed Dim | Heads | Parameters | Use Case |
|-------|------------|--------|-----------|-------|------------|----------|
| **ViT-S/16** | 16×16 | 12 | 384 | 6 | ~22M | Mobile/Edge, Fast inference |
| **ViT-S/32** | 32×32 | 12 | 384 | 6 | ~22M | Real-time applications |
| **ViT-B/16** | 16×16 | 12 | 768 | 12 | ~86M | General purpose, Transfer learning |
| **ViT-B/32** | 32×32 | 12 | 768 | 12 | ~86M | Balanced performance |
| **ViT-L/16** | 16×16 | 24 | 1024 | 16 | ~307M | High accuracy applications |
| **ViT-L/32** | 32×32 | 24 | 1024 | 16 | ~307M | Large-scale features |
| **ViT-H/14** | 14×14 | 32 | 1280 | 16 | ~632M | State-of-the-art performance |
| **ViT-H/16** | 16×16 | 32 | 1280 | 16 | ~632M | Research, Maximum capacity |

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

Requirements:
- TensorFlow >= 2.10.0
- NumPy >= 1.21.0

### Basic Usage

```python
from vit_all_variants import create_vit_b16

# Create a ViT-B/16 model for feature extraction
model = create_vit_b16(
    image_size=224,
    include_top=False  # No classification head
)

# Process an image
import tensorflow as tf
image = tf.random.normal((1, 224, 224, 3))
features = model(image, training=False)

# Output shape: (1, 197, 768)
# 197 tokens = 196 patches (14×14 grid) + 1 class token
# 768 = embedding dimension
```

## 📖 Usage Examples

### 1. Feature Extraction

Extract features for transfer learning, similarity search, or embeddings:

```python
from vit_all_variants import create_vit_b32

# Create feature extractor
model = create_vit_b32(image_size=224, include_top=False)

# Extract class token features (recommended for most tasks)
features = model.extract_features(images)
# Shape: (batch_size, 768)

# Or get all tokens (class token + patch tokens)
all_features = model(images, training=False)
# Shape: (batch_size, num_tokens, 768)
```

### 2. Image Classification

Add a classification head for supervised learning:

```python
from vit_all_variants import create_vit_b16

# Create model with classification head
model = create_vit_b16(
    image_size=224,
    include_top=True,
    num_classes=10  # e.g., CIFAR-10
)

# Compile and train
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train on your dataset
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

### 3. Transfer Learning with Frozen Backbone

Use pre-trained features with custom classification layers:

```python
from vit_all_variants import create_vit_b16
import tensorflow as tf

# Create base model and freeze it
base_model = create_vit_b16(image_size=224, include_top=False)
base_model.trainable = False

# Build custom model
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = x[:, 0, :]  # Extract class token
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

### 4. Using Different Variants

```python
from vit_all_variants import (
    create_vit_s16,  # Small/16
    create_vit_s32,  # Small/32
    create_vit_b16,  # Base/16
    create_vit_b32,  # Base/32
    create_vit_l16,  # Large/16
    create_vit_l32,  # Large/32
    create_vit_h14,  # Huge/14
    create_vit_h16,  # Huge/16
)

# Or use the generic factory
from vit_all_variants import create_vit

model = create_vit(
    variant="vit_l16",  # Specify variant
    image_size=224,
    include_top=False,
    dropout=0.1
)
```

### 5. Custom Image Sizes

ViT models work with any image size divisible by the patch size:

```python
from vit_all_variants import create_vit_b32

# ViT-B/32 works with image sizes divisible by 32
for img_size in [224, 256, 384, 512]:
    model = create_vit_b32(image_size=img_size, include_top=False)
    num_patches = (img_size // 32) ** 2
    print(f"Image size: {img_size}×{img_size}")
    print(f"Number of patches: {num_patches}")
    print(f"Total tokens: {num_patches + 1}")  # +1 for class token
```

### 6. Save and Load Models

```python
from vit_all_variants import create_vit_b16

# Create and save
model = create_vit_b16(image_size=224, include_top=False)
model.save_weights("vit_b16_weights")

# Load later
new_model = create_vit_b16(
    image_size=224,
    include_top=False,
    weights="vit_b16_weights"
)
```

## 🔍 Model Specifications

### Architecture Details

All models follow the standard Vision Transformer architecture:

1. **Patch Embedding**: Image → Patches → Linear Projection
2. **Position Embedding**: Learnable position embeddings added to patch embeddings
3. **Class Token**: Prepended learnable token for classification
4. **Transformer Encoder**: Stack of multi-head self-attention + MLP blocks
5. **Layer Normalization**: Applied before each sub-layer (Pre-LN)
6. **Classification Head**: Optional dense layer for supervised learning

### Parameter Counts

```
ViT-S variants:  ~22 million parameters
ViT-B variants:  ~86 million parameters
ViT-L variants:  ~307 million parameters
ViT-H variants:  ~632 million parameters
```

### Output Shapes

For image size 224×224:

| Model | Patch Grid | Num Patches | Total Tokens | Output Shape (batch=1) |
|-------|------------|-------------|--------------|------------------------|
| ViT-S/32 | 7×7 | 49 | 50 | (1, 50, 384) |
| ViT-B/32 | 7×7 | 49 | 50 | (1, 50, 768) |
| ViT-S/16 | 14×14 | 196 | 197 | (1, 197, 384) |
| ViT-B/16 | 14×14 | 196 | 197 | (1, 197, 768) |
| ViT-L/16 | 14×14 | 196 | 197 | (1, 197, 1024) |
| ViT-H/16 | 14×14 | 196 | 197 | (1, 197, 1280) |
| ViT-H/14 | 16×16 | 256 | 257 | (1, 257, 1280) |

## 🎯 Use Case Recommendations

### Real-time Applications (Mobile/Edge)
**Recommended:** ViT-S/32
**Reason:** Smallest model, fastest inference, suitable for resource-constrained devices

### Balanced Performance
**Recommended:** ViT-S/16, ViT-B/32
**Reason:** Good trade-off between speed and accuracy

### High Accuracy (Cloud/Server)
**Recommended:** ViT-B/16, ViT-L/16
**Reason:** Better feature extraction with reasonable compute requirements

### State-of-the-art Performance
**Recommended:** ViT-L/16, ViT-H/14
**Reason:** Highest accuracy, suitable for research and high-end applications

### Feature Extraction & Transfer Learning
**Recommended:** ViT-B/16, ViT-B/32
**Reason:** Standard choice for embeddings, widely used and well-supported

### Fine-tuning on Small Datasets
**Recommended:** ViT-S/16, ViT-S/32
**Reason:** Less prone to overfitting, faster training

### Large-scale Pre-training
**Recommended:** ViT-L/16, ViT-H/14
**Reason:** Better capacity for learning from massive datasets

## ⚡ Performance Comparison

### Speed vs Accuracy Trade-offs

**Patch Size Impact** (within same model size):
- Larger patches (32×32) → Faster inference, fewer tokens, slightly lower accuracy
- Smaller patches (16×16, 14×14) → Slower inference, more tokens, higher accuracy

**Model Size Impact**:
- Smaller models (S) → Faster, less memory, lower accuracy
- Larger models (B, L, H) → Slower, more memory, higher accuracy

### Expected ImageNet-1K Performance

Typical Top-1 accuracy on ImageNet-1K (with sufficient pre-training):

- ViT-S/32: ~75%
- ViT-S/16: ~78%
- ViT-B/32: ~75-77%
- ViT-B/16: ~80-82%
- ViT-L/16: ~83-85%
- ViT-H/14: ~86-88%

*Note: Actual performance depends on pre-training data, augmentation, and training recipes.*

## 🛠️ Advanced Usage

### Custom Configuration

```python
from vit_all_variants import VisionTransformer

# Create custom ViT with specific parameters
model = VisionTransformer(
    image_size=384,
    patch_size=16,
    num_layers=24,
    embed_dim=1024,
    num_heads=16,
    mlp_dim=4096,
    dropout=0.1,
    include_top=True,
    num_classes=1000
)

model.build((None, 384, 384, 3))
```

### List All Available Variants

```python
from vit_all_variants import list_variants, compare_variants

# Show detailed specifications
list_variants()

# Compare all variants side-by-side
compare_variants(image_size=224)
```

### Benchmarking

```python
from compare_all_variants import main

# Run comprehensive analysis including:
# - Parameter counts
# - Memory requirements
# - Inference speed benchmarks
# - Output shapes
# - Use case recommendations
main()
```

## 📁 Project Structure

```
vision_transformer_b32/
├── vit_all_variants.py          # Main implementation (all variants)
├── example_usage.py              # Usage examples
├── compare_all_variants.py       # Benchmarking and comparison tools
├── requirements.txt              # Dependencies
├── README.md                     # This file
├── README_ALL_VARIANTS.md        # Detailed variant documentation
├── README_VIT.md                 # Original ViT documentation
└── QUICK_REFERENCE.md            # Quick reference guide
```

## 🔑 Key Features

- ✅ **All Standard Variants**: S, B, L, H models with multiple patch sizes
- ✅ **Flexible Image Sizes**: Support any size divisible by patch size
- ✅ **Feature Extraction**: Easy access to embeddings and representations
- ✅ **Transfer Learning**: Pre-trained backbone support
- ✅ **Custom Models**: Build variants with custom configurations
- ✅ **Save/Load**: Full model serialization support
- ✅ **Production Ready**: Clean, tested, documented code
- ✅ **Type Hints**: Full type annotations for better IDE support

## 📚 API Reference

### Factory Functions

All factory functions accept the following parameters:

```python
def create_vit_*(
    image_size: int = 224,           # Input image size (must be divisible by patch_size)
    include_top: bool = False,       # Whether to include classification head
    num_classes: int = 1000,         # Number of classes (if include_top=True)
    dropout: float = 0.1,            # Dropout rate
    weights: str = None              # Path to pre-trained weights
) -> VisionTransformer
```

**Available functions:**
- `create_vit(variant, ...)` - Generic factory, specify variant name
- `create_vit_s16(...)` - ViT-Small/16
- `create_vit_s32(...)` - ViT-Small/32
- `create_vit_b16(...)` - ViT-Base/16
- `create_vit_b32(...)` - ViT-Base/32
- `create_vit_l16(...)` - ViT-Large/16
- `create_vit_l32(...)` - ViT-Large/32
- `create_vit_h14(...)` - ViT-Huge/14
- `create_vit_h16(...)` - ViT-Huge/16

### VisionTransformer Class

```python
class VisionTransformer(keras.Model):
    def call(self, x, training=False):
        """
        Forward pass.

        Args:
            x: Input images (batch_size, height, width, 3)
            training: Whether in training mode

        Returns:
            If include_top=True: Class logits (batch_size, num_classes)
            If include_top=False: All tokens (batch_size, num_tokens, embed_dim)
        """

    def extract_features(self, x, training=False):
        """
        Extract class token features.

        Args:
            x: Input images (batch_size, height, width, 3)
            training: Whether in training mode

        Returns:
            Class token embeddings (batch_size, embed_dim)
        """
```

## 🎓 Training Tips

### Pre-training from Scratch

Vision Transformers require significant data and compute for pre-training:

```python
# Recommended settings for pre-training
model = create_vit_b16(
    image_size=224,
    include_top=True,
    num_classes=1000,
    dropout=0.0  # No dropout during pre-training
)

# Use strong augmentation (RandAugment, MixUp, CutMix)
# Large batch sizes (>1024)
# Long training (300+ epochs)
# Warmup learning rate schedule
```

### Fine-tuning

For fine-tuning on downstream tasks:

```python
# Load pre-trained weights (when available)
model = create_vit_b16(
    image_size=224,
    include_top=True,
    num_classes=10,  # Your task
    dropout=0.1,
    weights="path/to/pretrained/weights"
)

# Use lower learning rate (1e-4 to 1e-5)
# Shorter training (10-50 epochs)
# Consider freezing early layers initially
```

## 🔬 Research & Citation

This implementation is based on:

```bibtex
@article{dosovitskiy2020image,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and others},
  journal={arXiv preprint arXiv:2010.11929},
  year={2020}
}
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Pre-trained weight loading from official sources
- Additional augmentation strategies
- Multi-GPU training support
- ONNX export support
- Quantization for mobile deployment

## 📄 License

This implementation is provided for educational and research purposes.

## 🔗 Resources

- [Original Paper](https://arxiv.org/abs/2010.11929)
- [Official Repository](https://github.com/google-research/vision_transformer)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Documentation](https://keras.io/)

## ❓ FAQ

**Q: Which variant should I use?**
A: Start with ViT-B/16 for general purposes. Use ViT-S models for speed, ViT-L/H for maximum accuracy.

**Q: Do I need pre-trained weights?**
A: Highly recommended for small datasets. ViT models typically require large-scale pre-training (ImageNet-21K or larger) to work well.

**Q: Can I use different image sizes?**
A: Yes! Any size divisible by the patch size works. Just specify `image_size` parameter.

**Q: How do I extract features for similarity search?**
A: Use `model.extract_features(images)` to get the class token embedding, which is ideal for similarity tasks.

**Q: What's the difference between /16 and /32?**
A: The number indicates patch size. /16 uses 16×16 patches (more tokens, higher accuracy, slower), /32 uses 32×32 patches (fewer tokens, faster, slightly lower accuracy).

**Q: Can I mix and match parameters?**
A: Yes! Use the `VisionTransformer` class directly to create custom configurations.

## 📞 Support

For issues, questions, or suggestions:
- Check the example files (`example_usage.py`, `compare_all_variants.py`)
- Review the documentation (`README_ALL_VARIANTS.md`, `QUICK_REFERENCE.md`)
- Open an issue on GitHub

---

**Happy coding with Vision Transformers!** 🚀
