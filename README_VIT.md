# Vision Transformer (ViT) B32 - TensorFlow 2 Implementation

A TensorFlow 2 implementation of Vision Transformer B32, similar to the model available on Kaggle. This implementation focuses on feature extraction and can be used for various computer vision tasks.

## Architecture

**ViT-B/32 Specifications:**
- Image size: 224×224 (configurable)
- Patch size: 32×32
- Number of layers: 12
- Hidden dimension: 768
- Number of attention heads: 12
- MLP dimension: 3072
- Number of patches: 49 (7×7 grid for 224×224 images)
- Total tokens: 50 (49 patches + 1 class token)

## Files

- `vit_b32_model.py` - Main model implementation
- `example_usage.py` - Comprehensive examples
- `requirements.txt` - Python dependencies

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Feature Extraction

```python
from vit_b32_model import create_vit_b32
import tensorflow as tf

# Create feature extractor
model = create_vit_b32(image_size=224, include_top=False)

# Extract features from images
images = tf.random.normal((4, 224, 224, 3))
features = model.extract_features(images)
# Output shape: (4, 768) - 768-dimensional feature vectors
```

### 2. Image Classification

```python
from vit_b32_model import create_vit_b32

# Create classifier
model = create_vit_b32(
    image_size=224,
    include_top=True,
    num_classes=10
)

# Get predictions
logits = model(images)
probabilities = tf.nn.softmax(logits)
```

### 3. Fine-tuning

```python
from vit_b32_model import create_vit_b32

# Create model
model = create_vit_b32(
    image_size=224,
    include_top=True,
    num_classes=5
)

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train
model.fit(train_dataset, epochs=10, validation_data=val_dataset)
```

### 4. Transfer Learning

```python
from vit_b32_model import create_vit_b32

# Load pre-trained base model
base_model = create_vit_b32(image_size=224, include_top=False)
base_model.trainable = False  # Freeze backbone

# Add custom head
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = x[:, 0, :]  # Extract class token
x = tf.keras.layers.Dense(256, activation='relu')(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
```

## Model Components

### PatchEmbedding
Splits input images into patches and projects them into embedding space using a convolutional layer.

### MultiHeadSelfAttention
Implements multi-head self-attention mechanism for capturing relationships between patches.

### TransformerBlock
Complete transformer encoder block with:
- Layer normalization
- Multi-head self-attention
- MLP (feed-forward network)
- Residual connections

### VisionTransformerB32
Main model class that combines all components:
- Patch embedding
- Positional embeddings
- Class token
- Stack of transformer blocks
- Optional classification head

## Use Cases

1. **Feature Extraction**: Extract 768-dimensional features for downstream tasks
2. **Image Classification**: Train or fine-tune for classification
3. **Transfer Learning**: Use pre-trained features with frozen backbone
4. **Image Similarity**: Compare images using extracted features
5. **Embedding Space Analysis**: Analyze learned representations

## Custom Image Sizes

The model supports any image size divisible by the patch size (32):

```python
# 256×256 images -> 64 patches
model_256 = create_vit_b32(image_size=256, include_top=False)

# 384×384 images -> 144 patches
model_384 = create_vit_b32(image_size=384, include_top=False)
```

## Running Examples

```bash
# Run all examples
python example_usage.py

# Run model creation test
python vit_b32_model.py
```

## Model Parameters

Total parameters: **86,415,592** (~86M)
- Patch embedding: 589,824
- Transformer blocks: 85,054,464
- Position embeddings: 38,400
- Class token: 768

## Key Features

- Pure TensorFlow 2 implementation
- Fully serializable with `get_config()`
- Support for both feature extraction and classification
- Configurable input sizes
- Easy to extend and customize
- Compatible with TensorFlow's training APIs

## Notes

- Input images should be normalized (e.g., 0-1 range or ImageNet normalization)
- For best results, use input size 224×224 or larger
- Patch size of 32 means fewer patches but faster inference compared to ViT-B/16
- The model can be used with `model.fit()`, custom training loops, or TensorFlow Serving

## References

- Original ViT Paper: "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)
- Kaggle Model: https://www.kaggle.com/models/spsayakpaul/vision-transformer/tensorFlow2/vit-b32-fe

## License

Free to use for research and commercial purposes.
