"""
Script to create and save all Vision Transformer model variants.
This script builds all 8 ViT variants and saves them as downloadable .keras files.
"""

import sys
import os
import tensorflow as tf
from pathlib import Path

# Add src to path
sys.path.append('src')

from vit_all_variants import (
    create_vit_s16, create_vit_s32,
    create_vit_b16, create_vit_b32,
    create_vit_l16, create_vit_l32,
    create_vit_h14, create_vit_h16
)

# Create output directory
output_dir = Path("models")
output_dir.mkdir(exist_ok=True)

print("=" * 80)
print("Creating and saving all Vision Transformer variants")
print("=" * 80)

# Define all variants
variants = {
    'vit_s16': (create_vit_s16, 'ViT-S/16 - Small with 16x16 patches'),
    'vit_s32': (create_vit_s32, 'ViT-S/32 - Small with 32x32 patches'),
    'vit_b16': (create_vit_b16, 'ViT-B/16 - Base with 16x16 patches'),
    'vit_b32': (create_vit_b32, 'ViT-B/32 - Base with 32x32 patches'),
    'vit_l16': (create_vit_l16, 'ViT-L/16 - Large with 16x16 patches'),
    'vit_l32': (create_vit_l32, 'ViT-L/32 - Large with 32x32 patches'),
    'vit_h14': (create_vit_h14, 'ViT-H/14 - Huge with 14x14 patches'),
    'vit_h16': (create_vit_h16, 'ViT-H/16 - Huge with 16x16 patches'),
}

image_size = 224
total_models = 0

for variant_name, (create_fn, description) in variants.items():
    print(f"\n{'=' * 80}")
    print(f"Processing: {description}")
    print(f"{'=' * 80}")

    # Create feature extraction model (no classification head)
    print(f"  Creating feature extraction model (include_top=False)...")
    model_features = create_fn(image_size=image_size, include_top=False)
    params = model_features.count_params()
    print(f"    Parameters: {params:,}")

    # Build the model by running a forward pass
    dummy_input = tf.random.normal((1, image_size, image_size, 3))
    _ = model_features(dummy_input, training=False)
    print(f"    Model built successfully")

    # Save feature extraction model
    feature_path = output_dir / f"{variant_name}_features.keras"
    model_features.save(feature_path)
    file_size_mb = feature_path.stat().st_size / (1024 * 1024)
    print(f"    [+] Saved: {feature_path} ({file_size_mb:.1f} MB)")
    total_models += 1

    # Save weights only (smaller file)
    weights_path = output_dir / f"{variant_name}_weights.weights.h5"
    model_features.save_weights(weights_path)
    file_size_mb = weights_path.stat().st_size / (1024 * 1024)
    print(f"    [+] Saved: {weights_path} ({file_size_mb:.1f} MB)")
    total_models += 1

    # Create classifier model (with classification head for 1000 classes like ImageNet)
    print(f"  Creating classification model (include_top=True)...")
    model_classifier = create_fn(
        image_size=image_size,
        include_top=True,
        num_classes=1000,
        dropout=0.1
    )
    params_clf = model_classifier.count_params()
    print(f"    Parameters: {params_clf:,}")

    # Build the classifier model
    _ = model_classifier(dummy_input, training=False)
    print(f"    Model built successfully")

    # Save classifier model
    classifier_path = output_dir / f"{variant_name}_classifier_1000.keras"
    model_classifier.save(classifier_path)
    file_size_mb = classifier_path.stat().st_size / (1024 * 1024)
    print(f"    [+] Saved: {classifier_path} ({file_size_mb:.1f} MB)")
    total_models += 1

    # Clean up to save memory
    del model_features, model_classifier
    tf.keras.backend.clear_session()

print(f"\n{'=' * 80}")
print(f"SUMMARY")
print(f"{'=' * 80}")
print(f"Total models created: {total_models}")
print(f"Output directory: {output_dir.absolute()}")
print(f"\nModel types saved for each variant:")
print(f"  1. *_features.keras - Feature extraction model (no classification head)")
print(f"  2. *_weights.weights.h5 - Weights only (smaller, for loading)")
print(f"  3. *_classifier_1000.keras - Full classifier for 1000 classes")
print(f"\n[SUCCESS] All models saved successfully!")
print(f"{'=' * 80}")

# Create a README in the models directory
readme_content = """# Vision Transformer Model Files

This directory contains all Vision Transformer model variants in multiple formats.

## File Types

For each variant (e.g., `vit_b16`), you'll find:

1. **`{variant}_features.keras`** - Feature extraction model (no classification head)
   - Use: Extract embeddings, transfer learning, similarity search
   - Output: Token embeddings (class token + patch tokens)

2. **`{variant}_weights.weights.h5`** - Weights only (smallest file size)
   - Use: Load into a model created with the same architecture
   - Requires: Create model first, then load weights

3. **`{variant}_classifier_1000.keras`** - Full classification model
   - Use: Direct classification (ImageNet-1K style with 1000 classes)
   - Output: Class logits

## Available Variants

### Small Models (Fast, ~22M parameters)
- `vit_s16` - Small with 16×16 patches (197 tokens)
- `vit_s32` - Small with 32×32 patches (50 tokens, fastest)

### Base Models (Balanced, ~86M parameters)
- `vit_b16` - Base with 16×16 patches (197 tokens, recommended)
- `vit_b32` - Base with 32×32 patches (50 tokens)

### Large Models (High accuracy, ~305M parameters)
- `vit_l16` - Large with 16×16 patches (197 tokens)
- `vit_l32` - Large with 32×32 patches (50 tokens)

### Huge Models (State-of-the-art, ~631M parameters)
- `vit_h14` - Huge with 14×14 patches (257 tokens, most detailed)
- `vit_h16` - Huge with 16×16 patches (197 tokens)

## Usage Examples

### Loading Feature Extraction Model

```python
import tensorflow as tf

# Load complete model
model = tf.keras.models.load_model('models/vit_b16_features.keras')

# Extract features
images = tf.random.normal((4, 224, 224, 3))
features = model.extract_features(images)  # Shape: (4, 768)
```

### Loading Weights Only

```python
from vit_all_variants import create_vit_b16

# Create model architecture
model = create_vit_b16(image_size=224, include_top=False)

# Load saved weights
model.load_weights('models/vit_b16_weights.weights.h5')
```

### Loading Classifier

```python
import tensorflow as tf

# Load classifier
model = tf.keras.models.load_model('models/vit_b16_classifier_1000.keras')

# Make predictions
images = tf.random.normal((1, 224, 224, 3))
logits = model(images, training=False)
probabilities = tf.nn.softmax(logits)
```

## Model Selection Guide

**By Use Case:**
- **General Purpose**: ViT-B/16
- **Speed Priority**: ViT-S/32 or ViT-B/32
- **Accuracy Priority**: ViT-L/16 or ViT-H/14
- **Resource Constrained**: ViT-S variants

**By GPU Memory:**
- **8GB**: ViT-S variants
- **16GB**: ViT-B variants
- **24GB**: ViT-L variants
- **40GB+**: ViT-H variants

## File Sizes

Approximate file sizes:
- Small variants: ~85-350 MB
- Base variants: ~340-1.3 GB
- Large variants: ~1.2-4.5 GB
- Huge variants: ~2.5-9.5 GB

Weights-only files (*.weights.h5) are typically smaller than full model files.

## Notes

- All models are initialized with random weights (not pre-trained on ImageNet)
- Image size: 224×224 (can be changed during model creation)
- All models are compatible with TensorFlow 2.10+
- Models use the architecture from "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)
"""

readme_path = output_dir / "README.md"
with open(readme_path, 'w') as f:
    f.write(readme_content)

print(f"\n[+] Created README: {readme_path}")
