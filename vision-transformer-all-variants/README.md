# Vision Transformer (ViT) - All Variants

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-FF6F00?logo=tensorflow)](https://tensorflow.org)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python)](https://python.org)
[![Kaggle](https://img.shields.io/badge/Kaggle-Model-20BEFF?logo=kaggle)](https://kaggle.com)

A production-ready TensorFlow/Keras implementation of Vision Transformer (ViT) models with all standard variants.

> **📄 Model Source:** Implementation based on ["An Image is Worth 16x16 Words"](https://arxiv.org/abs/2010.11929) (Dosovitskiy et al., 2021). Clean-room TensorFlow/Keras implementation following the original architecture specifications.

## 🚀 Quick Start on Kaggle

### Method 1: Direct Import (Recommended)

```python
import sys
sys.path.append('/kaggle/input/vision-transformer-all-variants/src')

from vit_all_variants import create_vit_b16

# Create model
model = create_vit_b16(image_size=224, include_top=False)

# Extract features
import tensorflow as tf
images = tf.random.normal((4, 224, 224, 3))
features = model(images, training=False)
print(f"Output shape: {features.shape}")  # (4, 197, 768)
```

### Method 2: Load from Kaggle Input

```python
# In your Kaggle notebook, add this dataset as input
# Then use the code directly
import sys
sys.path.append('/kaggle/input/vision-transformer-all-variants/src')

from vit_all_variants import (
    create_vit_s16, create_vit_s32,
    create_vit_b16, create_vit_b32,
    create_vit_l16, create_vit_l32,
    create_vit_h14, create_vit_h16
)
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

## 💡 Common Use Cases on Kaggle

### 1. Feature Extraction for Competition

```python
import sys
sys.path.append('/kaggle/input/vision-transformer-all-variants/src')
from vit_all_variants import create_vit_b16
import tensorflow as tf

# Load your competition data
train_images = ...  # Your training images

# Create feature extractor
model = create_vit_b16(image_size=224, include_top=False)

# Extract features
features = model.extract_features(train_images)
# Use features for your competition model
```

### 2. Transfer Learning

```python
import sys
sys.path.append('/kaggle/input/vision-transformer-all-variants/src')
from vit_all_variants import create_vit_b16
import tensorflow as tf

# Create base model
base_model = create_vit_b16(image_size=224, include_top=False)
base_model.trainable = False  # Freeze for faster training

# Add custom head
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = x[:, 0, :]  # Extract class token
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Train on competition data
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

### 3. Image Similarity Search

```python
import sys
sys.path.append('/kaggle/input/vision-transformer-all-variants/src')
from vit_all_variants import create_vit_b32
import tensorflow as tf
import numpy as np

# Create feature extractor (B/32 for speed)
model = create_vit_b32(image_size=224, include_top=False)

# Extract embeddings
query_features = model.extract_features(query_images)
database_features = model.extract_features(database_images)

# Compute cosine similarity
query_norm = tf.nn.l2_normalize(query_features, axis=1)
database_norm = tf.nn.l2_normalize(database_features, axis=1)
similarity = tf.matmul(query_norm, database_norm, transpose_b=True)

# Find most similar images
top_k_indices = tf.argsort(similarity, direction='DESCENDING')[:, :10]
```

### 4. Fine-tuning on Custom Dataset

```python
import sys
sys.path.append('/kaggle/input/vision-transformer-all-variants/src')
from vit_all_variants import create_vit_s16
import tensorflow as tf

# Use smaller model for faster experimentation
model = create_vit_s16(
    image_size=224,
    include_top=True,
    num_classes=10,  # Your dataset classes
    dropout=0.1
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20,
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(patience=3),
        tf.keras.callbacks.EarlyStopping(patience=5)
    ]
)
```

## 🎯 Choosing the Right Model

**For Kaggle Competitions:**
- **Quick Experiments**: Use ViT-S/32 or ViT-B/32 (faster iterations)
- **Feature Extraction**: Use ViT-B/16 (best balance)
- **Maximum Accuracy**: Use ViT-L/16 or ViT-H/14 (if GPU time allows)

**GPU Memory Considerations:**
- **Kaggle Free Tier (16GB)**: ViT-B/16 with batch size 16-32
- **Kaggle GPU P100**: ViT-L/16 with batch size 8-16
- **Multiple GPUs**: ViT-H/14 with larger batches

## 📊 Typical Performance

Expected Top-1 accuracy on ImageNet-1K (with pre-training):
- ViT-S/32: ~75%
- ViT-B/32: ~75-77%
- ViT-B/16: ~80-82%
- ViT-L/16: ~83-85%
- ViT-H/14: ~86-88%

## 🔧 Advanced Features

### Compare All Variants

```python
from vit_all_variants import compare_variants, list_variants

# See detailed comparison
compare_variants(image_size=224)

# List all specifications
list_variants()
```

### Custom Configuration

```python
from vit_all_variants import VisionTransformer

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
```

## 📁 Package Structure

```
vision-transformer-all-variants/
├── src/
│   ├── __init__.py
│   └── vit_all_variants.py      # Main implementation
├── examples/
│   └── example_usage.py          # Usage examples
├── requirements.txt              # Dependencies
├── dataset-metadata.json         # Kaggle metadata
└── README.md                     # This file
```

## 🔑 Key Features

- ✅ All 8 standard ViT variants (S/B/L/H with 14/16/32 patches)
- ✅ Easy feature extraction for transfer learning
- ✅ Flexible image sizes (any size divisible by patch size)
- ✅ Classification head support (optional)
- ✅ Save/load model weights
- ✅ Production-ready, well-documented code
- ✅ Optimized for Kaggle workflows

## 📚 Additional Resources

- **Original Paper**: [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- **Example Notebook**: See `examples/example_usage.py`
- **Full Documentation**: Check the source code docstrings

## 🎓 Citation

If you use this implementation in your work:

```bibtex
@article{dosovitskiy2020image,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and others},
  journal={arXiv preprint arXiv:2010.11929},
  year={2020}
}
```

## 💬 Tips for Kaggle Users

1. **Start Small**: Test with ViT-S/32 first, then scale up
2. **Feature Extraction**: Use `extract_features()` for quick embeddings
3. **Memory Management**: Use smaller batch sizes for larger models
4. **Mixed Precision**: Enable for faster training on Kaggle GPUs
5. **Save Checkpoints**: Use callbacks to save best models

## 🚀 Getting Started

1. Add this dataset to your Kaggle notebook
2. Copy the import code from Quick Start section
3. Start building your model!

---

**Happy Kaggling!** 🏆

For issues or questions, please use the Kaggle discussion tab.
