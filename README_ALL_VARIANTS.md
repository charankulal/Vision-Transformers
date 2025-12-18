# Vision Transformer - All Variants

Complete TensorFlow 2 implementation of all Vision Transformer (ViT) variants from the original paper "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020).

## Available Models

| Variant | Patch Size | Layers | Embed Dim | Heads | Parameters | Speed | Use Case |
|---------|-----------|--------|-----------|-------|------------|-------|----------|
| **ViT-S/16** | 16×16 | 12 | 384 | 6 | ~22M | ⚡⚡⚡ | Mobile, Fast inference |
| **ViT-S/32** | 32×32 | 12 | 384 | 6 | ~22M | ⚡⚡⚡⚡ | Real-time, Edge devices |
| **ViT-B/16** | 16×16 | 12 | 768 | 12 | ~86M | ⚡⚡ | Standard choice |
| **ViT-B/32** | 32×32 | 12 | 768 | 12 | ~86M | ⚡⚡⚡ | Fast feature extraction |
| **ViT-L/16** | 16×16 | 24 | 1024 | 16 | ~307M | ⚡ | High accuracy |
| **ViT-L/32** | 32×32 | 24 | 1024 | 16 | ~307M | ⚡⚡ | Large-scale training |
| **ViT-H/14** | 14×14 | 32 | 1280 | 16 | ~632M | 🐌 | State-of-the-art |
| **ViT-H/16** | 16×16 | 32 | 1280 | 16 | ~632M | 🐌 | Research, SOTA |

## Quick Start

### Installation

```bash
cd vision_transformer_b32
pip install -r requirements.txt
```

### Basic Usage

```python
from vit_all_variants import create_vit

# Create any variant by name
model = create_vit('vit_b16', image_size=224, include_top=False)

# Or use specific factory functions
from vit_all_variants import create_vit_s16, create_vit_b32, create_vit_l16

model_small = create_vit_s16(image_size=224)
model_base = create_vit_b32(image_size=224)
model_large = create_vit_l16(image_size=224)
```

## Detailed Examples

### 1. Feature Extraction with Different Variants

```python
import tensorflow as tf
from vit_all_variants import create_vit_s32, create_vit_b16, create_vit_l16

# Prepare images
images = tf.random.normal((8, 224, 224, 3))

# Fast feature extraction (ViT-S/32)
fast_model = create_vit_s32(image_size=224, include_top=False)
fast_features = fast_model.extract_features(images)
# Output: (8, 384) - 384-dimensional features

# Balanced feature extraction (ViT-B/16)
balanced_model = create_vit_b16(image_size=224, include_top=False)
balanced_features = balanced_model.extract_features(images)
# Output: (8, 768) - 768-dimensional features

# High-quality feature extraction (ViT-L/16)
powerful_model = create_vit_l16(image_size=224, include_top=False)
powerful_features = powerful_model.extract_features(images)
# Output: (8, 1024) - 1024-dimensional features
```

### 2. Classification with Different Model Sizes

```python
from vit_all_variants import create_vit

# Small model for fast inference
small_classifier = create_vit(
    'vit_s16',
    image_size=224,
    include_top=True,
    num_classes=10
)

# Base model for balanced performance
base_classifier = create_vit(
    'vit_b16',
    image_size=224,
    include_top=True,
    num_classes=10
)

# Large model for high accuracy
large_classifier = create_vit(
    'vit_l16',
    image_size=224,
    include_top=True,
    num_classes=10
)
```

### 3. Choosing the Right Model

```python
from vit_all_variants import create_vit

# For mobile/edge deployment → Use Small models
mobile_model = create_vit('vit_s32', image_size=224, include_top=False)

# For cloud deployment with good balance → Use Base models
cloud_model = create_vit('vit_b16', image_size=224, include_top=False)

# For maximum accuracy → Use Large/Huge models
sota_model = create_vit('vit_l16', image_size=224, include_top=False)
```

### 4. Custom Image Sizes

```python
from vit_all_variants import create_vit_b16

# All patch sizes must divide the image size evenly
model_224 = create_vit_b16(image_size=224)  # 14×14 patches
model_384 = create_vit_b16(image_size=384)  # 24×24 patches
model_512 = create_vit_b16(image_size=512)  # 32×32 patches
```

## Model Specifications

### ViT-Small (S)
- **Parameters**: ~22M
- **Embedding dimension**: 384
- **Layers**: 12
- **Attention heads**: 6
- **MLP dimension**: 1536
- **Best for**: Mobile apps, edge devices, real-time inference
- **Typical accuracy**: 75-78% on ImageNet

### ViT-Base (B)
- **Parameters**: ~86M
- **Embedding dimension**: 768
- **Layers**: 12
- **Attention heads**: 12
- **MLP dimension**: 3072
- **Best for**: Standard use, transfer learning, feature extraction
- **Typical accuracy**: 75-82% on ImageNet

### ViT-Large (L)
- **Parameters**: ~307M
- **Embedding dimension**: 1024
- **Layers**: 24
- **Attention heads**: 16
- **MLP dimension**: 4096
- **Best for**: High-accuracy applications, large datasets
- **Typical accuracy**: 83-85% on ImageNet

### ViT-Huge (H)
- **Parameters**: ~632M
- **Embedding dimension**: 1280
- **Layers**: 32
- **Attention heads**: 16
- **MLP dimension**: 5120
- **Best for**: State-of-the-art research, massive datasets
- **Typical accuracy**: 86-88% on ImageNet

## Patch Size Trade-offs

### Smaller Patches (/14, /16)
✅ **Advantages**:
- Higher accuracy
- Better fine-grained details
- More tokens for attention

❌ **Disadvantages**:
- Slower inference
- Higher memory usage
- More computation

### Larger Patches (/32)
✅ **Advantages**:
- Faster inference (4× faster than /16)
- Lower memory usage
- Suitable for resource-constrained devices

❌ **Disadvantages**:
- Slightly lower accuracy
- Less detailed features
- Fewer tokens

## Performance Comparison

### Inference Speed (224×224 images, relative)
1. **ViT-S/32**: 1.0× (baseline, fastest)
2. **ViT-S/16**: ~4×slower
3. **ViT-B/32**: ~4× slower
4. **ViT-B/16**: ~16× slower
5. **ViT-L/32**: ~15× slower
6. **ViT-L/16**: ~60× slower
7. **ViT-H/14**: ~100× slower
8. **ViT-H/16**: ~120× slower

### Number of Tokens (224×224 images)
- **Patch 32**: 49 patches + 1 class = **50 tokens**
- **Patch 16**: 196 patches + 1 class = **197 tokens**
- **Patch 14**: 256 patches + 1 class = **257 tokens**

## Use Case Recommendations

### Real-time Applications
```python
# Mobile apps, video processing, edge devices
model = create_vit('vit_s32', image_size=224, include_top=False)
```

### Balanced Production Use
```python
# Web services, batch processing, APIs
model = create_vit('vit_b16', image_size=224, include_top=False)
```

### High-Accuracy Requirements
```python
# Medical imaging, security, quality control
model = create_vit('vit_l16', image_size=224, include_top=False)
```

### Research & Experimentation
```python
# Pushing boundaries, academic research
model = create_vit('vit_h14', image_size=224, include_top=False)
```

## Running Benchmarks

Compare all variants:
```bash
python compare_all_variants.py
```

This will show:
- Detailed specifications
- Inference speed benchmarks
- Memory requirements
- Output shapes
- Use case recommendations

List available variants:
```bash
python vit_all_variants.py
```

## Advanced Usage

### Transfer Learning with Frozen Backbone

```python
from vit_all_variants import create_vit_b16

# Load base model
base_model = create_vit_b16(image_size=224, include_top=False)
base_model.trainable = False  # Freeze

# Add custom head
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = x[:, 0, :]  # Extract class token
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
```

### Progressive Training (Start Small, Then Scale)

```python
# 1. Start with small model for quick experimentation
small_model = create_vit('vit_s16', include_top=True, num_classes=10)
small_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
small_model.fit(train_data, epochs=5)

# 2. Move to base model for better accuracy
base_model = create_vit('vit_b16', include_top=True, num_classes=10)
base_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
base_model.fit(train_data, epochs=10)

# 3. Fine-tune with large model
large_model = create_vit('vit_l16', include_top=True, num_classes=10)
large_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
large_model.fit(train_data, epochs=20)
```

### Ensemble Different Variants

```python
from vit_all_variants import create_vit_s16, create_vit_b16, create_vit_l16

# Create ensemble of different sizes
model_s = create_vit_s16(include_top=True, num_classes=10)
model_b = create_vit_b16(include_top=True, num_classes=10)
model_l = create_vit_l16(include_top=True, num_classes=10)

# Get predictions from all models
pred_s = tf.nn.softmax(model_s(images))
pred_b = tf.nn.softmax(model_b(images))
pred_l = tf.nn.softmax(model_l(images))

# Average predictions
ensemble_pred = (pred_s + pred_b + pred_l) / 3
```

## Memory Requirements (Approximate)

| Model | Parameters | Model Size | Peak Memory (inference) |
|-------|-----------|------------|-------------------------|
| ViT-S/16 | 22M | ~88 MB | ~200 MB |
| ViT-S/32 | 22M | ~88 MB | ~150 MB |
| ViT-B/16 | 86M | ~345 MB | ~600 MB |
| ViT-B/32 | 86M | ~345 MB | ~400 MB |
| ViT-L/16 | 307M | ~1.2 GB | ~2 GB |
| ViT-L/32 | 307M | ~1.2 GB | ~1.5 GB |
| ViT-H/14 | 632M | ~2.5 GB | ~5 GB |
| ViT-H/16 | 632M | ~2.5 GB | ~4.5 GB |

## File Structure

```
vision_transformer_b32/
├── vit_b32_model.py          # Original B32 implementation
├── vit_all_variants.py        # All variants with factory functions
├── compare_all_variants.py    # Benchmark and comparison tool
├── example_usage.py           # Usage examples
├── requirements.txt           # Dependencies
├── README_VIT.md             # Original B32 documentation
└── README_ALL_VARIANTS.md    # This file
```

## Tips & Best Practices

1. **Start Small**: Begin with ViT-S/32 or ViT-S/16 for quick prototyping
2. **Scale Up**: Move to larger models once you've validated your approach
3. **Patch Size**: Use /32 for speed, /16 for accuracy, /14 for maximum performance
4. **Image Size**: Larger images (384, 512) improve accuracy but slow down inference
5. **Fine-tuning**: Freeze backbone for small datasets, train end-to-end for large ones
6. **Batch Size**: Smaller models allow larger batch sizes
7. **Mixed Precision**: Use `tf.keras.mixed_precision` to reduce memory and speed up training

## Common Issues

### Out of Memory
- Use smaller model variant (S instead of B, B instead of L)
- Use larger patch size (/32 instead of /16)
- Reduce image size
- Enable mixed precision training

### Slow Inference
- Use /32 patch size instead of /16
- Use smaller model (S or B)
- Reduce image size
- Enable XLA compilation

### Overfitting
- Use smaller model (ViT-S)
- Add more dropout
- Use data augmentation
- Reduce training epochs

## References

- Original Paper: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- Kaggle Model: https://www.kaggle.com/models/spsayakpaul/vision-transformer/
- Official Implementation: https://github.com/google-research/vision_transformer

## License

Free to use for research and commercial purposes.
