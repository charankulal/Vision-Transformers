"""Vision Transformer - Quick Usage Examples"""

import tensorflow as tf
from vit_all_variants import create_vit_b16, create_vit_b32

# ============================================================================
# EXAMPLE 1: Feature Extraction
# ============================================================================
print("Example 1: Feature Extraction")
model = create_vit_b16(image_size=224, include_top=False)
images = tf.random.normal((4, 224, 224, 3))
features = model.extract_features(images)  # (4, 768)
print(f"Features shape: {features.shape}\n")

# ============================================================================
# EXAMPLE 2: Image Classification
# ============================================================================
print("Example 2: Classification")
model = create_vit_b16(image_size=224, include_top=True, num_classes=10)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
print("Model ready for training\n")

# ============================================================================
# EXAMPLE 3: Transfer Learning
# ============================================================================
print("Example 3: Transfer Learning")
base_model = create_vit_b16(image_size=224, include_top=False)
base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = x[:, 0, :]  # Class token
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
print("Transfer learning model ready\n")

# ============================================================================
# EXAMPLE 4: Different Image Sizes
# ============================================================================
print("Example 4: Different Image Sizes")
for size in [224, 384, 512]:
    model = create_vit_b16(image_size=size, include_top=False)
    img = tf.random.normal((1, size, size, 3))
    out = model(img)
    print(f"{size}×{size} → {out.shape}")
print()

# ============================================================================
# EXAMPLE 5: Save & Load
# ============================================================================
print("Example 5: Save & Load")
model = create_vit_b16(image_size=224, include_top=False)
model.save_weights('vit_weights.h5')
model.load_weights('vit_weights.h5')
print("Model saved and loaded\n")

# ============================================================================
# EXAMPLE 6: All Variants
# ============================================================================
print("Example 6: All Variants")
from vit_all_variants import (
    create_vit_s16, create_vit_b16,
    create_vit_l16, create_vit_h16
)

models = {
    'ViT-S/16': create_vit_s16(image_size=224, include_top=False),
    'ViT-B/16': create_vit_b16(image_size=224, include_top=False),
    'ViT-L/16': create_vit_l16(image_size=224, include_top=False),
    'ViT-H/16': create_vit_h16(image_size=224, include_top=False),
}

for name, m in models.items():
    feats = m.extract_features(tf.random.normal((1, 224, 224, 3)))
    print(f"{name}: {feats.shape}")

print("\nAll examples complete!")
