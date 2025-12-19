# Vision Transformer (ViT) - All Variants

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-FF6F00?logo=tensorflow)](https://tensorflow.org)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python)](https://python.org)
[![Kaggle](https://img.shields.io/badge/Kaggle-Model-20BEFF?logo=kaggle)](https://kaggle.com)

A production-ready TensorFlow/Keras implementation of Vision Transformer (ViT) models with all standard variants.

> **📄 Model Source:** Implementation based on ["An Image is Worth 16x16 Words"](https://arxiv.org/abs/2010.11929) (Dosovitskiy et al., 2021). Clean-room TensorFlow/Keras implementation following the original architecture specifications.

## 🚀 Quick Start

### Step 1: Add Dataset to Kaggle Notebook

Add this dataset as input to your Kaggle notebook.

### Step 2: Import the Package

```python
import sys
import tensorflow as tf

# Add the ViT package to Python path
sys.path.append('/kaggle/input/vision-transformer-all-variants/src')

# Import ViT models
from vit_all_variants import (
    create_vit_s16, create_vit_s32,
    create_vit_b16, create_vit_b32,
    create_vit_l16, create_vit_l32,
    create_vit_h14, create_vit_h16,
    list_variants, compare_variants
)

print("✓ Vision Transformer package imported successfully!")
```

### Step 3: List Available Models

```python
# See all 8 variants with specifications
list_variants()
```

**Output:**
```
VIT_S16: 16x16 patches, 12 layers, 384 embed, 6 heads → 21.7M params
VIT_S32: 32x32 patches, 12 layers, 384 embed, 6 heads → 22.5M params
VIT_B16: 16x16 patches, 12 layers, 768 embed, 12 heads → 85.8M params
VIT_B32: 32x32 patches, 12 layers, 768 embed, 12 heads → 87.5M params
VIT_L16: 16x16 patches, 24 layers, 1024 embed, 16 heads → 303.3M params
VIT_L32: 32x32 patches, 24 layers, 1024 embed, 16 heads → 305.5M params
VIT_H14: 14x14 patches, 32 layers, 1280 embed, 16 heads → 630.8M params
VIT_H16: 16x16 patches, 32 layers, 1280 embed, 16 heads → 630.9M params
```

## 📦 Available Models

| Model | Patch Size | Parameters | Best For |
|-------|------------|------------|----------|
| **ViT-S/16** | 16×16 | ~22M | Mobile/Edge devices |
| **ViT-S/32** | 32×32 | ~22M | Real-time applications |
| **ViT-B/16** | 16×16 | ~86M | General purpose |
| **ViT-B/32** | 32×32 | ~86M | Balanced performance |
| **ViT-L/16** | 16×16 | ~307M | High accuracy |
| **ViT-L/32** | 32×32 | ~307M | Large-scale features |
| **ViT-H/14** | 14×14 | ~632M | State-of-the-art |
| **ViT-H/16** | 16×16 | ~632M | Maximum capacity |

## 💡 Usage Examples

### Example 1: Feature Extraction

Extract features from images for downstream tasks.

```python
# Create ViT-B/16 for feature extraction
model = create_vit_b16(image_size=224, include_top=False)
print(f"Parameters: {model.count_params():,}")  # 85,798,656

# Extract features
images = tf.random.normal((4, 224, 224, 3))

# All tokens (includes class token + patch tokens)
all_features = model(images, training=False)
print(f"All tokens: {all_features.shape}")  # (4, 197, 768)

# Class token only (recommended for classification/similarity)
cls_features = model.extract_features(images, training=False)
print(f"Class token: {cls_features.shape}")  # (4, 768)
```

**Use Cases:**
- Image similarity search
- Transfer learning
- Clustering
- Downstream classification

### Example 2: Image Classification

Train a model with classification head.

```python
# Create classifier
model = create_vit_b32(
    image_size=224,
    include_top=True,
    num_classes=10,
    dropout=0.1
)

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train
model.fit(train_dataset, validation_data=val_dataset, epochs=10)

# Predict
predictions = model(test_images, training=False)
probabilities = tf.nn.softmax(predictions)
```

### Example 3: Transfer Learning (Frozen Backbone)

Best approach for limited data or quick training.

```python
# Create and freeze base model
base_model = create_vit_b16(image_size=224, include_top=False)
base_model.trainable = False

# Add custom head
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = x[:, 0, :]  # Extract class token
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

# Create and compile model
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Train (only custom head is trainable)
model.fit(train_dataset, epochs=10)
```

**Result:** Only ~200K trainable parameters vs 85.8M total

### Example 4: Compare Model Variants

Choose the right model for your use case.

```python
variants = {
    'ViT-S/32': create_vit_s32,
    'ViT-B/16': create_vit_b16,
}

for name, create_fn in variants.items():
    model = create_fn(image_size=224, include_top=False)
    output = model(tf.random.normal((1, 224, 224, 3)))
    print(f"{name}: {model.count_params():,} params → {output.shape}")
```

**Output:**
```
ViT-S/32: 22,493,952 params → (1, 50, 384)
ViT-B/16: 85,798,656 params → (1, 197, 768)
```

### Example 5: Save and Load Models

```python
# Save weights
model = create_vit_s16(image_size=224, include_top=False)
model.save_weights('vit_s16_weights.weights.h5')

# Load weights
new_model = create_vit_s16(
    image_size=224,
    include_top=False,
    weights='vit_s16_weights.weights.h5'
)

# Verify they're identical
test_image = tf.random.normal((1, 224, 224, 3))
diff = tf.reduce_max(tf.abs(model(test_image) - new_model(test_image)))
print(f"Difference: {diff.numpy()}")  # 0.0
```

## 🎯 Model Selection Guide

**By Use Case:**
- **Kaggle/General Purpose**: ViT-B/16 (best balance)
- **Quick Experiments**: ViT-S/32 or ViT-B/32 (faster)
- **High Accuracy**: ViT-L/16 or ViT-H/14
- **Mobile/Edge**: ViT-S/32 (smallest, fastest)

**By GPU Memory:**
- **8GB GPU**: ViT-S variants
- **16GB GPU**: ViT-B/16 (batch 16-32)
- **24GB GPU**: ViT-L/16 (batch 8-16)
- **40GB+ GPU**: ViT-H variants (batch 4-8)

**By Speed vs Accuracy:**
```
Speed:    S/32 > S/16 > B/32 > B/16 > L/32 > L/16 > H/16 > H/14
Accuracy: S/32 < B/32 < S/16 < B/16 < L/32 < L/16 < H/16 < H/14
```

## 📊 Performance Reference

**ImageNet-1K Accuracy** (with pre-training):
- ViT-S/32: ~75% | ViT-S/16: ~78%
- ViT-B/32: 75-77% | ViT-B/16: 80-82%
- ViT-L/32: 77-79% | ViT-L/16: 83-85%
- ViT-H/14: 86-88% | ViT-H/16: 85-87%

## 🔧 Advanced Features

```python
# Compare all variants side-by-side
compare_variants(image_size=224)

# List all variant specifications
list_variants()

# Use different image sizes (must be divisible by patch size)
model_224 = create_vit_b16(image_size=224)
model_384 = create_vit_b16(image_size=384)
model_512 = create_vit_b16(image_size=512)
```

## 📁 Repository Structure

```
vision-transformer-all-variants/
├── src/
│   ├── __init__.py
│   └── vit_all_variants.py         # Main implementation
├── examples/
│   ├── example_usage.py             # Usage examples
│   └── kaggle_quickstart.ipynb      # Interactive notebook
├── MODEL_INSTANCES.md               # Detailed model guide
├── MODEL_CARD.md                    # Full documentation
└── README.md                        # This file
```

## 📚 Documentation

- **Quick Reference**: See [MODEL_INSTANCES.md](MODEL_INSTANCES.md)
- **Original Paper**: ["An Image is Worth 16x16 Words"](https://arxiv.org/abs/2010.11929)
- **Example Notebook**: [kaggle_quickstart.ipynb](examples/kaggle_quickstart.ipynb)
- **Full Model Card**: [MODEL_CARD.md](MODEL_CARD.md)

## 💡 Tips & Best Practices

1. **Start with ViT-B/16** - Best all-around model
2. **Use `extract_features()`** - Get class token for classification/similarity
3. **Freeze backbone first** - Train only custom head, then fine-tune if needed
4. **Smaller models for small datasets** - Avoid overfitting with ViT-S variants
5. **Higher resolution for details** - Use 384×384 or 512×512 for fine-grained tasks
6. **Save checkpoints** - Use `ModelCheckpoint` callback during training

## 🎓 Citation

```bibtex
@article{dosovitskiy2020image,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and others},
  journal={arXiv preprint arXiv:2010.11929},
  year={2020}
}
```

---

**Ready to use?** Check out [kaggle_quickstart.ipynb](examples/kaggle_quickstart.ipynb) for a complete walkthrough!

For questions or issues, please use the Kaggle discussion tab.
