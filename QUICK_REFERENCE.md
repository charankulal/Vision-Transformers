# Vision Transformer - Quick Reference Guide

## Model Selection Flowchart

```
Need a Vision Transformer model?
│
├─ Real-time inference needed? (Mobile/Edge)
│  └─ YES → ViT-S/32 ⚡⚡⚡⚡
│
├─ Want best speed/accuracy balance?
│  ├─ Speed priority → ViT-B/32 ⚡⚡⚡
│  └─ Accuracy priority → ViT-B/16 ⚡⚡
│
├─ Need high accuracy? (Cloud deployment)
│  ├─ Speed still matters → ViT-L/32 ⚡⚡
│  └─ Accuracy is key → ViT-L/16 ⚡
│
└─ Need state-of-the-art? (Research/SOTA)
   ├─ Maximum performance → ViT-H/14 🐌
   └─ Slightly faster → ViT-H/16 🐌
```

## Quick Command Reference

```python
from vit_all_variants import create_vit

# Replace 'vit_XXX' with your choice:
# vit_s16, vit_s32, vit_b16, vit_b32, vit_l16, vit_l32, vit_h14, vit_h16

model = create_vit('vit_b16', image_size=224, include_top=False)
```

## At-a-Glance Comparison

| Model | Params | Speed | Accuracy | Memory | Use When... |
|-------|--------|-------|----------|--------|-------------|
| **S/32** | 22M | ⚡⚡⚡⚡ | ⭐⭐⭐ | 88MB | Mobile, real-time, low resources |
| **S/16** | 22M | ⚡⚡⚡ | ⭐⭐⭐⭐ | 88MB | Mobile with better accuracy |
| **B/32** | 86M | ⚡⚡⚡ | ⭐⭐⭐⭐ | 345MB | Fast feature extraction |
| **B/16** | 86M | ⚡⚡ | ⭐⭐⭐⭐⭐ | 345MB | **Standard choice** ✨ |
| **L/32** | 307M | ⚡⚡ | ⭐⭐⭐⭐⭐ | 1.2GB | Large-scale with speed needs |
| **L/16** | 307M | ⚡ | ⭐⭐⭐⭐⭐⭐ | 1.2GB | High accuracy applications |
| **H/14** | 632M | 🐌 | ⭐⭐⭐⭐⭐⭐⭐ | 2.5GB | Research, SOTA performance |
| **H/16** | 632M | 🐌 | ⭐⭐⭐⭐⭐⭐⭐ | 2.5GB | Research with faster inference |

## Common Scenarios

### Scenario 1: Mobile App
```python
model = create_vit('vit_s32', image_size=224, include_top=False)
# Why: Fastest, smallest, runs on phones
```

### Scenario 2: Web API
```python
model = create_vit('vit_b32', image_size=224, include_top=False)
# Why: Good balance, handles traffic well
```

### Scenario 3: Batch Processing
```python
model = create_vit('vit_b16', image_size=224, include_top=False)
# Why: Better accuracy, speed less critical
```

### Scenario 4: Medical Imaging
```python
model = create_vit('vit_l16', image_size=384, include_top=False)
# Why: High accuracy crucial, resources available
```

### Scenario 5: Research Paper
```python
model = create_vit('vit_h14', image_size=384, include_top=False)
# Why: SOTA results, benchmark comparison
```

## Code Snippets

### Feature Extraction (Most Common)
```python
from vit_all_variants import create_vit_b16

model = create_vit_b16(image_size=224, include_top=False)
features = model.extract_features(images)  # (batch, 768)
```

### Classification
```python
from vit_all_variants import create_vit_b16

model = create_vit_b16(
    image_size=224,
    include_top=True,
    num_classes=10
)
logits = model(images)
predictions = tf.nn.softmax(logits)
```

### Transfer Learning
```python
from vit_all_variants import create_vit_b16

base = create_vit_b16(image_size=224, include_top=False)
base.trainable = False

inputs = tf.keras.Input((224, 224, 3))
features = base(inputs, training=False)[:, 0, :]
outputs = tf.keras.layers.Dense(num_classes)(features)
model = tf.keras.Model(inputs, outputs)
```

## Performance Numbers

### Inference Time (Single Image, CPU)
- ViT-S/32: ~50ms
- ViT-S/16: ~200ms
- ViT-B/32: ~200ms
- ViT-B/16: ~800ms
- ViT-L/16: ~3000ms
- ViT-H/14: ~5000ms

### Typical ImageNet Accuracy
- ViT-S/32: 75%
- ViT-S/16: 78%
- ViT-B/32: 77%
- ViT-B/16: 81%
- ViT-L/16: 84%
- ViT-H/14: 87%

## Decision Matrix

| Your Constraint | Recommended Model |
|----------------|-------------------|
| < 100 MB model size | ViT-S/XX |
| < 500 MB model size | ViT-B/XX |
| < 50ms inference | ViT-S/32 |
| < 200ms inference | ViT-S/16 or ViT-B/32 |
| < 1s inference | ViT-B/16 or ViT-L/32 |
| Need > 80% accuracy | ViT-B/16 or larger |
| Need > 85% accuracy | ViT-L/16 or ViT-H/XX |
| Mobile deployment | ViT-S/32 |
| Edge devices | ViT-S/32 or ViT-S/16 |
| Cloud deployment | ViT-B/16 or ViT-L/16 |
| GPU available | Any model |
| CPU only | ViT-S/XX or ViT-B/32 |

## Patch Size Decoder

**What does /32, /16, /14 mean?**

For 224×224 image:
- **/32**: Divides into 7×7 = **49 patches** (4× faster, slightly less accurate)
- **/16**: Divides into 14×14 = **196 patches** (balanced)
- **/14**: Divides into 16×16 = **256 patches** (most accurate, slowest)

**Rule of thumb**: Smaller patch number = faster but less detailed

## Model Size Decoder

**What does S, B, L, H mean?**

- **S (Small)**: 384-dim embeddings, 22M params
- **B (Base)**: 768-dim embeddings, 86M params ← Most popular
- **L (Large)**: 1024-dim embeddings, 307M params
- **H (Huge)**: 1280-dim embeddings, 632M params

## Quick Benchmark Command

```bash
# See detailed comparison of all models
python compare_all_variants.py

# List all available variants
python vit_all_variants.py
```

## FAQ

**Q: Which model should I start with?**
A: ViT-B/16 - it's the most balanced and widely used.

**Q: I need it fast, which one?**
A: ViT-S/32 or ViT-B/32

**Q: I need it accurate, which one?**
A: ViT-L/16 or ViT-H/14

**Q: What's the difference between /16 and /32?**
A: /32 is 4× faster, /16 is more accurate

**Q: Can I use different image sizes?**
A: Yes! Just make sure size is divisible by patch size

**Q: Which for transfer learning?**
A: ViT-B/16 is the standard choice

**Q: Out of memory error?**
A: Use smaller model (S instead of B) or larger patches (/32 instead of /16)

## Getting Help

1. Check `README_ALL_VARIANTS.md` for detailed documentation
2. Run `python vit_all_variants.py` to see all configs
3. Run `python compare_all_variants.py` for benchmarks
4. See `example_usage.py` for code examples

## Common Import Patterns

```python
# Generic creation
from vit_all_variants import create_vit
model = create_vit('vit_b16')

# Specific variants
from vit_all_variants import (
    create_vit_s16, create_vit_s32,
    create_vit_b16, create_vit_b32,
    create_vit_l16, create_vit_l32,
    create_vit_h14, create_vit_h16
)

# Use configs directly
from vit_all_variants import VIT_CONFIGS
print(VIT_CONFIGS['vit_b16'])
```

## Pro Tips

1. **Start small**: Prototype with ViT-S/32, then scale up
2. **Batch size**: Larger models need smaller batches
3. **Image size**: 224 is standard, 384 is high-res, 512 is overkill
4. **Mixed precision**: Use `tf.keras.mixed_precision` to save memory
5. **Compilation**: Enable XLA for faster inference
6. **Freezing**: Freeze backbone for small datasets (< 10K images)

---

**Need more details?** See `README_ALL_VARIANTS.md`
**Want examples?** See `example_usage.py`
**Need comparisons?** Run `python compare_all_variants.py`
