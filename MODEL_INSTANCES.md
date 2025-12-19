# Vision Transformer Model Instances Documentation

Complete documentation for all Vision Transformer variants with model-specific descriptions, overviews, and usage examples.

---

## ViT-S/16 (Small, 16×16 patches)

### Model Description

ViT-S/16 is a compact Vision Transformer variant designed for resource-constrained environments while maintaining good accuracy. With approximately 22 million parameters and a 16×16 patch size, it processes 224×224 images into 197 tokens (196 spatial patches + 1 class token). This model is ideal for mobile deployment, edge computing, and scenarios requiring fast iteration during development.

**Key Specifications:**
- **Parameters**: ~22M
- **Layers**: 12 transformer blocks
- **Embedding Dimension**: 384
- **Attention Heads**: 6
- **MLP Dimension**: 1536
- **Patch Size**: 16×16 pixels
- **Expected Accuracy**: ~78% on ImageNet-1K (with pre-training)

### Overview

**Best For:**
- Mobile and edge device deployment
- Fast experimentation and prototyping
- Fine-tuning on small to medium datasets (10K-100K images)
- Real-time inference requirements with GPU
- Educational purposes and research with limited compute

**Advantages:**
- Lower computational cost than base models
- Faster training and inference
- Less prone to overfitting on small datasets
- Suitable for deployment on consumer GPUs (GTX 1660+)

**Limitations:**
- Lower accuracy ceiling compared to larger models
- May struggle with very fine-grained classification
- Requires GPU for acceptable inference speed

**Performance Characteristics:**
- Inference latency: ~15ms per image (V100 GPU)
- Memory requirement: 2GB GPU RAM
- Training time: ~7 days for ImageNet-21K pre-training (single GPU)
- FLOPs: 4.6G per forward pass

### Example Usage

#### Basic Feature Extraction

```python
import sys
sys.path.append('/kaggle/input/vision-transformer-all-variants/src')
from vit_all_variants import create_vit_s16
import tensorflow as tf

# Create ViT-S/16 feature extractor
model = create_vit_s16(
    image_size=224,
    include_top=False,
    dropout=0.1
)

# Load and preprocess images
images = tf.random.normal((8, 224, 224, 3))  # Batch of 8 images

# Extract features
features = model(images, training=False)
print(f"Feature shape: {features.shape}")  # (8, 197, 384)

# Extract class token only (for classification/similarity)
class_tokens = model.extract_features(images)
print(f"Class token shape: {class_tokens.shape}")  # (8, 384)
```

#### Fine-tuning for Custom Classification

```python
from vit_all_variants import create_vit_s16
import tensorflow as tf

# Create model with classification head
model = create_vit_s16(
    image_size=224,
    include_top=True,
    num_classes=10,  # e.g., CIFAR-10
    dropout=0.2  # Higher dropout for small datasets
)

# Compile with appropriate optimizer
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy', 'top_k_categorical_accuracy']
)

# Training callbacks
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'vit_s16_best.h5',
        save_best_only=True,
        monitor='val_accuracy'
    )
]

# Train
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=30,
    callbacks=callbacks
)
```

#### Transfer Learning for Kaggle Competition

```python
from vit_all_variants import create_vit_s16
import tensorflow as tf

# Create frozen base model
base_model = create_vit_s16(image_size=224, include_top=False)
base_model.trainable = False

# Build custom classifier
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)

# Use class token
x = x[:, 0, :]  # Shape: (batch, 384)

# Add custom layers
x = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile and train
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_data, validation_data=val_data, epochs=20)

# Optional: Unfreeze for fine-tuning
base_model.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),  # Lower learning rate
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(train_data, validation_data=val_data, epochs=10)
```

#### Image Similarity Search

```python
from vit_all_variants import create_vit_s16
import tensorflow as tf
import numpy as np

# Create feature extractor
encoder = create_vit_s16(image_size=224, include_top=False)

def compute_embeddings(images):
    """Extract normalized embeddings for similarity search."""
    features = encoder.extract_features(images)
    # L2 normalization for cosine similarity
    return tf.nn.l2_normalize(features, axis=1)

# Index your database
database_embeddings = compute_embeddings(database_images)

# Query
query_embedding = compute_embeddings(query_image)

# Compute cosine similarity
similarity_scores = tf.matmul(query_embedding, database_embeddings, transpose_b=True)

# Get top-k similar images
top_k = 10
top_k_indices = tf.argsort(similarity_scores[0], direction='DESCENDING')[:top_k]
print(f"Most similar images: {top_k_indices.numpy()}")
```

---

## ViT-S/32 (Small, 32×32 patches)

### Model Description

ViT-S/32 is the fastest Vision Transformer variant, using larger 32×32 patches to reduce the number of tokens and computational cost. With the same 22 million parameters as ViT-S/16 but processing only 50 tokens instead of 197, this model excels in real-time applications and scenarios where speed is critical. It's the recommended choice for initial prototyping and mobile deployment.

**Key Specifications:**
- **Parameters**: ~22M
- **Layers**: 12 transformer blocks
- **Embedding Dimension**: 384
- **Attention Heads**: 6
- **MLP Dimension**: 1536
- **Patch Size**: 32×32 pixels
- **Expected Accuracy**: ~75% on ImageNet-1K (with pre-training)

### Overview

**Best For:**
- Real-time inference applications
- Mobile and embedded systems
- Quick prototyping and experimentation
- Batch processing large image collections
- Scenarios where speed trumps accuracy

**Advantages:**
- Fastest inference among all ViT models
- Lowest memory footprint
- 3× faster than ViT-S/16 due to fewer tokens
- Can run on CPU for offline processing
- Excellent throughput for batch operations

**Limitations:**
- Lower accuracy than smaller patch sizes (~3% vs ViT-S/16)
- Less spatial detail captured
- May miss fine-grained features

**Performance Characteristics:**
- Inference latency: ~8ms per image (V100 GPU)
- Memory requirement: 2GB GPU RAM
- Training time: ~5 days for ImageNet-21K pre-training (single GPU)
- FLOPs: 1.4G per forward pass
- Throughput: ~125 images/second (V100, batch size 32)

### Example Usage

#### Real-time Video Processing

```python
from vit_all_variants import create_vit_s32
import tensorflow as tf
import cv2

# Create fast feature extractor
model = create_vit_s32(image_size=224, include_top=False)

# Video capture
cap = cv2.VideoCapture('input_video.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)

# Process video frames
batch = []
batch_size = 8

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (224, 224))
    frame_normalized = frame_resized / 255.0

    batch.append(frame_normalized)

    # Process batch
    if len(batch) == batch_size:
        frames_tensor = tf.constant(batch, dtype=tf.float32)
        features = model.extract_features(frames_tensor)

        # Use features for downstream task (classification, tracking, etc.)
        # ... your application logic ...

        batch = []

cap.release()
```

#### Lightweight Image Classifier

```python
from vit_all_variants import create_vit_s32
import tensorflow as tf

# Create lightweight classifier
model = create_vit_s32(
    image_size=224,
    include_top=True,
    num_classes=100,
    dropout=0.15
)

# Compile for training
model.compile(
    optimizer=tf.keras.optimizers.Adam(3e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Mixed precision for faster training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Train with data augmentation
train_data = train_dataset.map(lambda x, y: (
    tf.image.random_flip_left_right(x),
    y
)).batch(64).prefetch(tf.data.AUTOTUNE)

model.fit(
    train_data,
    validation_data=val_dataset.batch(64),
    epochs=25
)

# Save for deployment
model.save('vit_s32_classifier.h5')
```

#### Fast Batch Feature Extraction

```python
from vit_all_variants import create_vit_s32
import tensorflow as tf
import numpy as np

# Create model
encoder = create_vit_s32(image_size=224, include_top=False)

def extract_features_fast(image_paths, batch_size=32):
    """Extract features from large image collection efficiently."""
    all_features = []

    # Create dataset pipeline
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)

    def load_and_preprocess(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [224, 224])
        img = img / 255.0
        return img

    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Extract features
    for batch in dataset:
        features = encoder.extract_features(batch)
        all_features.append(features.numpy())

    return np.vstack(all_features)

# Use it
image_list = ['img1.jpg', 'img2.jpg', ...]  # Your images
features = extract_features_fast(image_list, batch_size=64)
print(f"Extracted features shape: {features.shape}")  # (num_images, 384)
```

#### Mobile Deployment Preparation

```python
from vit_all_variants import create_vit_s32
import tensorflow as tf

# Create model
model = create_vit_s32(
    image_size=224,
    include_top=True,
    num_classes=10
)

# Load trained weights
model.load_weights('vit_s32_trained.h5')

# Convert to TensorFlow Lite for mobile
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optimize for mobile
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

# Convert
tflite_model = converter.convert()

# Save
with open('vit_s32_mobile.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model ready for mobile deployment!")
```

---

## ViT-B/16 (Base, 16×16 patches)

### Model Description

ViT-B/16 is the most widely used Vision Transformer variant, offering an excellent balance between accuracy and computational efficiency. With 86 million parameters and 16×16 patches, it's the standard choice for transfer learning, feature extraction, and general-purpose computer vision tasks. This model has been extensively validated across diverse datasets and is the recommended starting point for most applications.

**Key Specifications:**
- **Parameters**: ~86M
- **Layers**: 12 transformer blocks
- **Embedding Dimension**: 768
- **Attention Heads**: 12
- **MLP Dimension**: 3072
- **Patch Size**: 16×16 pixels
- **Expected Accuracy**: ~80-82% on ImageNet-1K (with pre-training)

### Overview

**Best For:**
- General-purpose image classification
- Transfer learning foundation
- Feature extraction for downstream tasks
- Kaggle competitions and research
- Production deployments with GPU availability
- Benchmark comparisons

**Advantages:**
- Industry-standard architecture with extensive community support
- Best accuracy-to-compute ratio
- Wide availability of pre-trained weights
- Well-studied performance characteristics
- Compatible with most GPU configurations

**Limitations:**
- Requires GPU for efficient inference
- Higher memory footprint than Small variants
- Pre-training from scratch is computationally expensive

**Performance Characteristics:**
- Inference latency: ~35ms per image (V100 GPU)
- Memory requirement: 4GB GPU RAM
- Training time: ~14 days for ImageNet-21K pre-training (single GPU)
- FLOPs: 17.6G per forward pass
- Throughput: ~30 images/second (V100, batch size 16)

### Example Usage

#### Standard Transfer Learning Pipeline

```python
from vit_all_variants import create_vit_b16
import tensorflow as tf

# Create base model (assume pre-trained weights available)
base_model = create_vit_b16(
    image_size=224,
    include_top=False,
    weights='path/to/pretrained_weights.h5'  # Load pre-trained weights
)

# Freeze base model
base_model.trainable = False

# Build complete model
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = x[:, 0, :]  # Class token

# Task-specific head
x = tf.keras.layers.LayerNormalization()(x)
x = tf.keras.layers.Dense(512, activation='gelu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(num_classes)(x)
outputs = tf.keras.layers.Activation('softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Phase 1: Train only the head
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', 'top_k_categorical_accuracy']
)

history1 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10
)

# Phase 2: Fine-tune entire model
base_model.trainable = True

# Use lower learning rate for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', 'top_k_categorical_accuracy']
)

history2 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10
)
```

#### Feature Extraction for Downstream Tasks

```python
from vit_all_variants import create_vit_b16
import tensorflow as tf
import numpy as np

# Create feature extractor
feature_extractor = create_vit_b16(image_size=224, include_top=False)

def extract_features(images):
    """Extract 768-dimensional features from images."""
    features = feature_extractor.extract_features(images)
    return features.numpy()

# Extract features for your dataset
train_features = extract_features(train_images)
test_features = extract_features(test_images)

# Use features with classical ML algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Train a simple classifier
clf = LogisticRegression(max_iter=1000, C=1.0)
clf.fit(train_features, train_labels)

# Evaluate
accuracy = clf.score(test_features, test_labels)
print(f"Accuracy with ViT-B/16 features: {accuracy:.2%}")
```

#### Multi-resolution Feature Pyramid

```python
from vit_all_variants import create_vit_b16
import tensorflow as tf

class MultiScaleViT(tf.keras.Model):
    """Extract features at multiple scales for object detection."""

    def __init__(self):
        super().__init__()
        self.vit_small = create_vit_b16(image_size=224, include_top=False)
        self.vit_large = create_vit_b16(image_size=448, include_top=False)

    def call(self, inputs, training=False):
        # Small scale (faster, coarse features)
        small_img = tf.image.resize(inputs, [224, 224])
        small_features = self.vit_small.extract_features(small_img, training=training)

        # Large scale (slower, fine features)
        large_img = tf.image.resize(inputs, [448, 448])
        large_features = self.vit_large.extract_features(large_img, training=training)

        # Concatenate multi-scale features
        combined = tf.concat([small_features, large_features], axis=-1)
        return combined

# Use multi-scale model
model = MultiScaleViT()
images = tf.random.normal((4, 512, 512, 3))
features = model(images)
print(f"Multi-scale features: {features.shape}")  # (4, 1536)
```

#### Self-supervised Contrastive Learning

```python
from vit_all_variants import create_vit_b16
import tensorflow as tf

class ContrastiveViT(tf.keras.Model):
    """ViT with projection head for contrastive learning (SimCLR-style)."""

    def __init__(self, projection_dim=128):
        super().__init__()
        self.encoder = create_vit_b16(image_size=224, include_top=False)

        # Projection head
        self.projection = tf.keras.Sequential([
            tf.keras.layers.Dense(2048, activation='relu'),
            tf.keras.layers.Dense(projection_dim)
        ])

    def call(self, inputs, training=False):
        features = self.encoder.extract_features(inputs, training=training)
        projections = self.projection(features, training=training)
        # L2 normalize for cosine similarity
        return tf.nn.l2_normalize(projections, axis=1)

# Contrastive loss (NT-Xent)
def contrastive_loss(projections_1, projections_2, temperature=0.5):
    batch_size = tf.shape(projections_1)[0]

    # Compute similarity matrix
    projections = tf.concat([projections_1, projections_2], axis=0)
    similarity_matrix = tf.matmul(projections, projections, transpose_b=True)
    similarity_matrix = similarity_matrix / temperature

    # Create labels (positive pairs)
    labels = tf.range(batch_size)
    labels = tf.concat([labels + batch_size, labels], axis=0)

    # Compute loss
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        labels,
        similarity_matrix,
        from_logits=True
    )

    return tf.reduce_mean(loss)

# Training loop
model = ContrastiveViT()
optimizer = tf.keras.optimizers.Adam(1e-4)

for images in unlabeled_dataset:
    # Create two augmented views
    view1 = augment(images)
    view2 = augment(images)

    with tf.GradientTape() as tape:
        proj1 = model(view1, training=True)
        proj2 = model(view2, training=True)
        loss = contrastive_loss(proj1, proj2)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

#### Production Deployment with TensorFlow Serving

```python
from vit_all_variants import create_vit_b16
import tensorflow as tf

# Create and train model
model = create_vit_b16(
    image_size=224,
    include_top=True,
    num_classes=1000
)

# ... training code ...

# Save for TensorFlow Serving
tf.saved_model.save(model, 'vit_b16_serving/1/')

# Client code for inference
import requests
import json

def predict_image(image_path):
    # Preprocess
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = img / 255.0

    # Prepare request
    data = json.dumps({
        "signature_name": "serving_default",
        "instances": img.numpy().tolist()
    })

    # Send to TensorFlow Serving
    response = requests.post(
        'http://localhost:8501/v1/models/vit_b16:predict',
        data=data
    )

    predictions = json.loads(response.text)['predictions']
    return predictions

# Use it
result = predict_image('test_image.jpg')
print(f"Top prediction: {result[0].argmax()}")
```

---

## ViT-B/32 (Base, 32×32 patches)

### Model Description

ViT-B/32 offers a balanced compromise between the accuracy of ViT-B/16 and the speed of smaller models. With 86 million parameters but only 50 tokens per image, it provides 2× faster inference than ViT-B/16 while maintaining ~75-77% accuracy on ImageNet. This variant is ideal for applications requiring good accuracy with moderate computational budgets.

**Key Specifications:**
- **Parameters**: ~86M
- **Layers**: 12 transformer blocks
- **Embedding Dimension**: 768
- **Attention Heads**: 12
- **MLP Dimension**: 3072
- **Patch Size**: 32×32 pixels
- **Expected Accuracy**: ~75-77% on ImageNet-1K (with pre-training)

### Overview

**Best For:**
- Balanced accuracy/speed requirements
- Medium-scale datasets (100K-1M images)
- Resource-efficient production deployments
- Batch processing pipelines
- Cloud deployments with cost constraints

**Advantages:**
- 2× faster inference than ViT-B/16
- Lower memory footprint
- Good accuracy for most practical applications
- Easier to fine-tune than larger models
- Better throughput for batch processing

**Limitations:**
- 3-5% lower accuracy than ViT-B/16
- May miss fine-grained spatial details
- Less effective for tasks requiring high spatial resolution

**Performance Characteristics:**
- Inference latency: ~18ms per image (V100 GPU)
- Memory requirement: 3GB GPU RAM
- Training time: ~10 days for ImageNet-21K pre-training (single GPU)
- FLOPs: 4.4G per forward pass
- Throughput: ~55 images/second (V100, batch size 32)

### Example Usage

#### Efficient Kaggle Competition Baseline

```python
from vit_all_variants import create_vit_b32
import tensorflow as tf

# Quick baseline for competitions
model = create_vit_b32(
    image_size=224,
    include_top=True,
    num_classes=num_competition_classes,
    dropout=0.2
)

# Mixed precision for faster training
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Use data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1)
])

# Training with augmentation
augmented_train = train_dataset.map(
    lambda x, y: (data_augmentation(x, training=True), y)
)

history = model.fit(
    augmented_train,
    validation_data=val_dataset,
    epochs=30,
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=3),
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=7),
        tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
    ]
)
```

#### Cost-Efficient Cloud Deployment

```python
from vit_all_variants import create_vit_b32
import tensorflow as tf

# Create optimized model for cloud deployment
model = create_vit_b32(image_size=224, include_top=True, num_classes=100)
model.load_weights('trained_weights.h5')

# Optimize for inference
@tf.function(input_signature=[tf.TensorSpec(shape=[None, 224, 224, 3], dtype=tf.float32)])
def optimized_predict(images):
    return model(images, training=False)

# Save optimized model
tf.saved_model.save(
    model,
    'vit_b32_optimized',
    signatures={'serving_default': optimized_predict}
)

# Benchmark inference cost
import time
import numpy as np

batch_sizes = [1, 4, 8, 16, 32]
for bs in batch_sizes:
    dummy_input = tf.random.normal((bs, 224, 224, 3))

    # Warmup
    for _ in range(10):
        _ = optimized_predict(dummy_input)

    # Measure
    start = time.time()
    iterations = 100
    for _ in range(iterations):
        _ = optimized_predict(dummy_input)
    end = time.time()

    avg_time = (end - start) / iterations
    throughput = bs / avg_time

    print(f"Batch size {bs}: {avg_time*1000:.2f}ms, {throughput:.1f} imgs/sec")
```

#### Content Moderation System

```python
from vit_all_variants import create_vit_b32
import tensorflow as tf

class ContentModerator:
    """Real-time content moderation using ViT-B/32."""

    def __init__(self, model_path, threshold=0.9):
        self.model = create_vit_b32(image_size=224, include_top=True, num_classes=2)
        self.model.load_weights(model_path)
        self.threshold = threshold

        # Categories: 0=safe, 1=unsafe

    def preprocess(self, image_path):
        """Load and preprocess image."""
        img = tf.io.read_file(image_path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [224, 224])
        img = img / 255.0
        return tf.expand_dims(img, 0)

    def moderate(self, image_path):
        """Check if image is safe for platform."""
        img = self.preprocess(image_path)
        logits = self.model(img, training=False)
        probs = tf.nn.softmax(logits)

        safe_prob = probs[0, 0].numpy()
        unsafe_prob = probs[0, 1].numpy()

        decision = {
            'safe': safe_prob > self.threshold,
            'confidence': float(max(safe_prob, unsafe_prob)),
            'safe_probability': float(safe_prob),
            'unsafe_probability': float(unsafe_prob),
            'requires_review': 0.4 < unsafe_prob < 0.6  # Uncertain cases
        }

        return decision

    def batch_moderate(self, image_paths, batch_size=32):
        """Process multiple images efficiently."""
        results = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_imgs = tf.stack([self.preprocess(p)[0] for p in batch_paths])

            logits = self.model(batch_imgs, training=False)
            probs = tf.nn.softmax(logits)

            for j, path in enumerate(batch_paths):
                safe_prob = probs[j, 0].numpy()
                unsafe_prob = probs[j, 1].numpy()

                results.append({
                    'image': path,
                    'safe': safe_prob > self.threshold,
                    'confidence': float(max(safe_prob, unsafe_prob))
                })

        return results

# Use it
moderator = ContentModerator('content_moderation_model.h5', threshold=0.85)

# Single image
result = moderator.moderate('user_upload.jpg')
if not result['safe']:
    print("⚠️ Content flagged for review")

# Batch processing
uploaded_images = ['img1.jpg', 'img2.jpg', 'img3.jpg', ...]
batch_results = moderator.batch_moderate(uploaded_images)
flagged = [r for r in batch_results if not r['safe']]
print(f"Flagged {len(flagged)}/{len(batch_results)} images")
```

#### Image Retrieval System

```python
from vit_all_variants import create_vit_b32
import tensorflow as tf
import numpy as np
from sklearn.neighbors import NearestNeighbors

class ImageRetrieval:
    """Fast image retrieval using ViT-B/32 embeddings."""

    def __init__(self):
        self.encoder = create_vit_b32(image_size=224, include_top=False)
        self.index = None
        self.image_paths = []

    def extract_embedding(self, image):
        """Extract normalized 768-dim embedding."""
        if isinstance(image, str):
            img = tf.io.read_file(image)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [224, 224])
            img = img / 255.0
            img = tf.expand_dims(img, 0)
        else:
            img = image

        embedding = self.encoder.extract_features(img)
        return tf.nn.l2_normalize(embedding, axis=1).numpy()

    def index_images(self, image_paths, batch_size=32):
        """Build search index from image collection."""
        print(f"Indexing {len(image_paths)} images...")
        embeddings = []

        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i+batch_size]
            batch_embeddings = []

            for path in batch:
                emb = self.extract_embedding(path)
                batch_embeddings.append(emb[0])

            embeddings.extend(batch_embeddings)

            if (i // batch_size) % 10 == 0:
                print(f"Processed {i}/{len(image_paths)} images")

        self.embeddings = np.array(embeddings)
        self.image_paths = image_paths

        # Build k-NN index
        self.index = NearestNeighbors(
            n_neighbors=20,
            metric='cosine',
            algorithm='brute'
        )
        self.index.fit(self.embeddings)

        print("✓ Indexing complete!")

    def search(self, query_image, top_k=10):
        """Find similar images."""
        query_embedding = self.extract_embedding(query_image)

        distances, indices = self.index.kneighbors(
            query_embedding,
            n_neighbors=top_k
        )

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append({
                'image_path': self.image_paths[idx],
                'similarity': 1 - dist,  # Convert distance to similarity
                'distance': float(dist)
            })

        return results

    def save_index(self, path):
        """Save index for later use."""
        np.savez(
            path,
            embeddings=self.embeddings,
            image_paths=self.image_paths
        )

    def load_index(self, path):
        """Load pre-built index."""
        data = np.load(path, allow_pickle=True)
        self.embeddings = data['embeddings']
        self.image_paths = data['image_paths'].tolist()

        self.index = NearestNeighbors(n_neighbors=20, metric='cosine')
        self.index.fit(self.embeddings)

# Usage example
retrieval = ImageRetrieval()

# Build index
database_images = ['img1.jpg', 'img2.jpg', ...]  # Your image collection
retrieval.index_images(database_images)
retrieval.save_index('image_index.npz')

# Search
query = 'query_image.jpg'
similar_images = retrieval.search(query, top_k=10)

for i, result in enumerate(similar_images, 1):
    print(f"{i}. {result['image_path']} (similarity: {result['similarity']:.3f})")
```

---

## ViT-L/16 (Large, 16×16 patches)

### Model Description

ViT-L/16 is a high-capacity Vision Transformer with 307 million parameters and 24 transformer layers. This model delivers state-of-the-art accuracy on standard benchmarks and excels in scenarios requiring maximum feature richness and representational power. With 1024-dimensional embeddings and 16 attention heads, ViT-L/16 is the go-to choice for research, high-accuracy production systems, and applications where accuracy is paramount.

**Key Specifications:**
- **Parameters**: ~307M
- **Layers**: 24 transformer blocks
- **Embedding Dimension**: 1024
- **Attention Heads**: 16
- **MLP Dimension**: 4096
- **Patch Size**: 16×16 pixels
- **Expected Accuracy**: ~83-85% on ImageNet-1K (with pre-training)

### Overview

**Best For:**
- State-of-the-art accuracy requirements
- Research and academic applications
- High-value production systems (medical, autonomous driving)
- Feature extraction for complex tasks
- Transfer learning with rich representations
- Fine-grained classification problems

**Advantages:**
- Highest accuracy among practical models
- Rich 1024-dimensional feature representations
- Excellent transfer learning performance
- Strong performance on fine-grained tasks
- Better generalization to out-of-distribution data

**Limitations:**
- Requires 24GB+ GPU for training
- Slower inference (85ms per image on V100)
- Higher deployment costs
- Longer training times
- Risk of overfitting on small datasets

**Performance Characteristics:**
- Inference latency: ~85ms per image (V100 GPU), ~51ms (A100 GPU)
- Memory requirement: 8GB GPU RAM for inference, 24GB+ for training
- Training time: ~21 days for ImageNet-21K pre-training (single GPU)
- FLOPs: 61.6G per forward pass
- Throughput: ~12 images/second (V100, batch size 8)

### Example Usage

#### High-Accuracy Transfer Learning

```python
from vit_all_variants import create_vit_l16
import tensorflow as tf

# Load large model with pre-trained weights
base_model = create_vit_l16(
    image_size=224,
    include_top=False,
    weights='path/to/imagenet21k_pretrained_weights.h5'
)

# Freeze for initial training
base_model.trainable = False

# Build model with custom head
inputs = tf.keras.Input(shape=(224, 224, 3))

# Extract features
x = base_model(inputs, training=False)
class_token = x[:, 0, :]  # Shape: (batch, 1024)

# Rich classification head
x = tf.keras.layers.LayerNormalization()(class_token)
x = tf.keras.layers.Dense(2048, activation='gelu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(1024, activation='gelu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(512, activation='gelu')(x)
x = tf.keras.layers.Dropout(0.1)(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Train with careful learning rate schedule
initial_learning_rate = 1e-3
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate,
    decay_steps=num_train_steps,
    alpha=1e-6
)

model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=0.01),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', 'top_k_categorical_accuracy']
)

# Phase 1: Train head only
history_head = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint('vit_l16_head.h5', save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir='logs/phase1')
    ]
)

# Phase 2: Fine-tune with very low learning rate
base_model.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-5, weight_decay=0.01),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', 'top_k_categorical_accuracy']
)

history_finetune = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint('vit_l16_finetuned.h5', save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
        tf.keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True)
    ]
)

print(f"Final validation accuracy: {max(history_finetune.history['val_accuracy']):.4f}")
```

#### Medical Image Classification

```python
from vit_all_variants import create_vit_l16
import tensorflow as tf

class MedicalImageClassifier:
    """High-accuracy medical image classifier using ViT-L/16."""

    def __init__(self, num_classes, image_size=384):
        """
        Args:
            num_classes: Number of diagnostic categories
            image_size: Higher resolution for medical images (384 or 512)
        """
        self.image_size = image_size

        # Create model with higher resolution
        self.model = create_vit_l16(
            image_size=image_size,
            include_top=True,
            num_classes=num_classes,
            dropout=0.2
        )

    def train(self, train_data, val_data, epochs=50):
        """Train with medical-image-specific augmentation."""

        # Medical-appropriate augmentation (no horizontal flips for some organs)
        augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.05),  # Small rotations
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.2),
            tf.keras.layers.RandomBrightness(0.1)
        ])

        augmented_train = train_data.map(
            lambda x, y: (augmentation(x, training=True), y)
        )

        # Use class weights for imbalanced medical datasets
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(train_labels),
            y=train_labels
        )
        class_weight_dict = dict(enumerate(class_weights))

        # Compile with appropriate loss
        self.model.compile(
            optimizer=tf.keras.optimizers.AdamW(1e-4, weight_decay=0.01),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )

        # Train with callbacks
        history = self.model.fit(
            augmented_train,
            validation_data=val_data,
            epochs=epochs,
            class_weight=class_weight_dict,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    'medical_vit_l16_best.h5',
                    monitor='val_auc',
                    save_best_only=True,
                    mode='max'
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_auc',
                    patience=10,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7
                )
            ]
        )

        return history

    def predict_with_uncertainty(self, images, num_samples=10):
        """Monte Carlo Dropout for uncertainty estimation."""
        predictions = []

        for _ in range(num_samples):
            pred = self.model(images, training=True)  # Enable dropout
            predictions.append(tf.nn.softmax(pred))

        predictions = tf.stack(predictions)

        # Mean prediction
        mean_pred = tf.reduce_mean(predictions, axis=0)

        # Uncertainty (entropy)
        entropy = -tf.reduce_sum(
            mean_pred * tf.math.log(mean_pred + 1e-10),
            axis=1
        )

        return {
            'predictions': mean_pred.numpy(),
            'uncertainty': entropy.numpy(),
            'std': tf.math.reduce_std(predictions, axis=0).numpy()
        }

# Usage
classifier = MedicalImageClassifier(num_classes=5, image_size=384)

# Train
history = classifier.train(
    train_dataset,
    val_dataset,
    epochs=50
)

# Inference with uncertainty
test_images = load_test_images()
results = classifier.predict_with_uncertainty(test_images)

for i, (pred, uncertainty) in enumerate(zip(results['predictions'], results['uncertainty'])):
    predicted_class = pred.argmax()
    confidence = pred.max()

    print(f"Image {i}:")
    print(f"  Predicted: Class {predicted_class}")
    print(f"  Confidence: {confidence:.3f}")
    print(f"  Uncertainty: {uncertainty:.3f}")

    if uncertainty > 0.5:  # High uncertainty threshold
        print("  ⚠️ High uncertainty - recommend human review")
```

#### Fine-Grained Recognition

```python
from vit_all_variants import create_vit_l16
import tensorflow as tf

# Fine-grained classification (e.g., bird species, dog breeds)
model = create_vit_l16(
    image_size=384,  # Higher resolution for fine details
    include_top=True,
    num_classes=200,  # e.g., CUB-200 bird species
    dropout=0.25
)

# Load pre-trained weights
model.load_weights('imagenet21k_vit_l16.h5', by_name=True, skip_mismatch=True)

# Compile with label smoothing for better generalization
model.compile(
    optimizer=tf.keras.optimizers.AdamW(1e-4, weight_decay=0.05),
    loss=tf.keras.losses.CategoricalCrossentropy(
        from_logits=True,
        label_smoothing=0.1
    ),
    metrics=['accuracy', 'top_k_categorical_accuracy']
)

# Advanced augmentation for fine-grained recognition
import tensorflow_addons as tfa

def advanced_augment(image, label):
    """Augmentation preserving fine-grained details."""
    # Random crop with high coverage
    image = tf.image.random_crop(
        tf.image.resize(image, [420, 420]),
        size=[384, 384, 3]
    )

    # Color augmentation
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_saturation(image, 0.8, 1.2)
    image = tf.image.random_hue(image, 0.1)

    # Geometric augmentation (careful with fine-grained features)
    image = tf.image.random_flip_left_right(image)
    image = tfa.image.rotate(image, tf.random.uniform([], -0.1, 0.1))

    # Normalization
    image = image / 255.0

    return image, label

# Apply augmentation
augmented_dataset = train_dataset.map(
    advanced_augment,
    num_parallel_calls=tf.data.AUTOTUNE
).batch(16).prefetch(tf.data.AUTOTUNE)

# Train
history = model.fit(
    augmented_dataset,
    validation_data=val_dataset.batch(16),
    epochs=100,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            'finegrained_vit_l16.h5',
            save_best_only=True,
            monitor='val_top_k_categorical_accuracy'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=5
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_top_k_categorical_accuracy',
            patience=15,
            restore_best_weights=True
        )
    ]
)
```

#### Research: Attention Visualization

```python
from vit_all_variants import create_vit_l16
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class AttentionVisualizer:
    """Visualize attention maps from ViT-L/16."""

    def __init__(self, model_weights=None):
        self.model = create_vit_l16(image_size=224, include_top=False)
        if model_weights:
            self.model.load_weights(model_weights)

    def extract_attention_maps(self, image, layer_idx=-1):
        """
        Extract attention maps from a specific transformer layer.

        Args:
            image: Input image (224, 224, 3)
            layer_idx: Which transformer layer to visualize (-1 for last)

        Returns:
            Attention weights: (num_heads, num_patches+1, num_patches+1)
        """
        # Get the transformer layer
        transformer_layers = [
            layer for layer in self.model.layers
            if 'transformer_block' in layer.name
        ]

        target_layer = transformer_layers[layer_idx]

        # Create a model that outputs attention weights
        attention_model = tf.keras.Model(
            inputs=self.model.input,
            outputs=target_layer.get_layer('mha').attention.output
        )

        # Forward pass
        image_batch = tf.expand_dims(image, 0)
        attention_weights = attention_model(image_batch, training=False)

        return attention_weights[0]  # Remove batch dimension

    def visualize_attention(self, image, head_idx=0, layer_idx=-1, patch_size=16):
        """Visualize attention from class token to patches."""

        attention_maps = self.extract_attention_maps(image, layer_idx)

        # Get attention from class token to all patches
        # Shape: (num_heads, num_tokens, num_tokens)
        class_attention = attention_maps[head_idx, 0, 1:]  # Skip class token

        # Reshape to spatial grid
        num_patches = int(np.sqrt(len(class_attention)))
        attention_grid = class_attention.numpy().reshape(num_patches, num_patches)

        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Original image
        axes[0].imshow(image.numpy())
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Attention map
        im = axes[1].imshow(attention_grid, cmap='hot', interpolation='bilinear')
        axes[1].set_title(f'Attention Map (Head {head_idx}, Layer {layer_idx})')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])

        plt.tight_layout()
        return fig

    def compare_all_heads(self, image, layer_idx=-1):
        """Visualize attention from all 16 heads."""

        attention_maps = self.extract_attention_maps(image, layer_idx)
        num_heads = attention_maps.shape[0]

        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        axes = axes.flatten()

        for head_idx in range(num_heads):
            class_attention = attention_maps[head_idx, 0, 1:]
            num_patches = int(np.sqrt(len(class_attention)))
            attention_grid = class_attention.numpy().reshape(num_patches, num_patches)

            axes[head_idx].imshow(attention_grid, cmap='hot', interpolation='bilinear')
            axes[head_idx].set_title(f'Head {head_idx}')
            axes[head_idx].axis('off')

        plt.suptitle(f'All Attention Heads (Layer {layer_idx})', fontsize=16)
        plt.tight_layout()
        return fig

# Usage
visualizer = AttentionVisualizer('vit_l16_pretrained.h5')

# Load test image
image = tf.io.read_file('test_image.jpg')
image = tf.image.decode_jpeg(image, channels=3)
image = tf.image.resize(image, [224, 224])
image = image / 255.0

# Visualize single head
fig1 = visualizer.visualize_attention(image, head_idx=5, layer_idx=-1)
plt.savefig('attention_head5.png', dpi=300, bbox_inches='tight')

# Compare all heads
fig2 = visualizer.compare_all_heads(image, layer_idx=-1)
plt.savefig('all_attention_heads.png', dpi=300, bbox_inches='tight')

plt.show()
```

---

## ViT-L/32 (Large, 32×32 patches)

### Model Description

ViT-L/32 combines the high capacity of the Large architecture (307M parameters, 24 layers, 1024-dim embeddings) with the efficiency of 32×32 patches. This creates a unique model that offers strong representational power with moderate computational requirements—faster than ViT-L/16 while maintaining higher capacity than ViT-B variants. It's particularly effective for tasks requiring rich features but not necessarily fine spatial resolution.

**Key Specifications:**
- **Parameters**: ~307M
- **Layers**: 24 transformer blocks
- **Embedding Dimension**: 1024
- **Attention Heads**: 16
- **MLP Dimension**: 4096
- **Patch Size**: 32×32 pixels
- **Expected Accuracy**: ~77-79% on ImageNet-1K (with pre-training)

### Overview

**Best For:**
- High-capacity feature extraction
- Semantic understanding tasks
- Scene recognition and classification
- Transfer learning to diverse domains
- Applications requiring rich representations but with speed constraints
- Large-scale image retrieval systems

**Advantages:**
- Rich 1024-dimensional embeddings
- 3× faster than ViT-L/16
- Better features than ViT-B/32 for semantic tasks
- Excellent for transfer learning
- Good balance of capacity and efficiency

**Limitations:**
- 4-6% lower accuracy than ViT-L/16 on fine-grained tasks
- Still requires substantial GPU resources
- May miss fine spatial details
- Higher deployment costs than Base models

**Performance Characteristics:**
- Inference latency: ~45ms per image (V100 GPU), ~27ms (A100 GPU)
- Memory requirement: 6GB GPU RAM for inference, 20GB+ for training
- Training time: ~18 days for ImageNet-21K pre-training (single GPU)
- FLOPs: 15.4G per forward pass
- Throughput: ~22 images/second (V100, batch size 16)

### Example Usage

#### Large-Scale Image Retrieval

```python
from vit_all_variants import create_vit_l32
import tensorflow as tf
import numpy as np
import faiss  # For efficient similarity search

class LargeScaleRetrieval:
    """Efficient retrieval system for millions of images using ViT-L/32."""

    def __init__(self):
        # Use L/32 for balance of quality and speed
        self.encoder = create_vit_l32(image_size=224, include_top=False)
        self.index = None
        self.image_ids = []

    def extract_features(self, images):
        """Extract 1024-dim features."""
        features = self.encoder.extract_features(images)
        # L2 normalize for cosine similarity
        features = tf.nn.l2_normalize(features, axis=1)
        return features.numpy().astype('float32')

    def build_index(self, image_paths, batch_size=32, use_gpu=True):
        """Build FAISS index for fast search over millions of images."""

        print(f"Building index for {len(image_paths)} images...")

        # Extract all features
        all_features = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]

            # Load and preprocess batch
            batch_images = []
            for path in batch_paths:
                img = tf.io.read_file(path)
                img = tf.image.decode_jpeg(img, channels=3)
                img = tf.image.resize(img, [224, 224])
                img = img / 255.0
                batch_images.append(img)

            batch_tensor = tf.stack(batch_images)
            features = self.extract_features(batch_tensor)
            all_features.append(features)

            if i % 1000 == 0:
                print(f"Processed {i}/{len(image_paths)} images")

        # Concatenate all features
        all_features = np.vstack(all_features)

        # Build FAISS index
        dimension = 1024

        if use_gpu:
            # GPU index for faster search
            res = faiss.StandardGpuResources()
            index = faiss.GpuIndexFlatIP(res, dimension)
        else:
            # CPU index with IVF for efficiency
            nlist = 100  # Number of clusters
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            index.train(all_features)

        index.add(all_features)

        self.index = index
        self.image_ids = image_paths

        print(f"✓ Index built with {len(image_paths)} images")

    def search(self, query_image, top_k=100):
        """Search for similar images.

        Args:
            query_image: Query image (path or tensor)
            top_k: Number of results to return

        Returns:
            List of (image_id, similarity_score) tuples
        """
        # Extract query features
        if isinstance(query_image, str):
            img = tf.io.read_file(query_image)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [224, 224])
            img = img / 255.0
            img = tf.expand_dims(img, 0)
        else:
            img = query_image

        query_features = self.extract_features(img)

        # Search
        distances, indices = self.index.search(query_features, top_k)

        # Format results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append({
                'image_id': self.image_ids[idx],
                'similarity': float(dist)
            })

        return results

    def save_index(self, path):
        """Save index to disk."""
        faiss.write_index(self.index, f"{path}.faiss")
        np.save(f"{path}_ids.npy", np.array(self.image_ids, dtype=object))

    def load_index(self, path, use_gpu=True):
        """Load pre-built index."""
        self.index = faiss.read_index(f"{path}.faiss")

        if use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        self.image_ids = np.load(f"{path}_ids.npy", allow_pickle=True).tolist()

# Usage example
retrieval = LargeScaleRetrieval()

# Build index from 1 million images
million_images = load_image_paths()  # Your dataset
retrieval.build_index(million_images, batch_size=64)
retrieval.save_index('vit_l32_index_1m')

# Fast search
query = 'query_image.jpg'
results = retrieval.search(query, top_k=100)

print(f"Top 10 similar images:")
for i, res in enumerate(results[:10], 1):
    print(f"{i}. {res['image_id']} (similarity: {res['similarity']:.4f})")
```

#### Scene Understanding System

```python
from vit_all_variants import create_vit_l32
import tensorflow as tf

class SceneUnderstanding:
    """Multi-task scene analysis using ViT-L/32."""

    def __init__(self, num_scenes=365, num_attributes=102):
        # Shared ViT-L/32 backbone
        self.backbone = create_vit_l32(image_size=224, include_top=False)

        # Build multi-task model
        inputs = tf.keras.Input(shape=(224, 224, 3))
        features = self.backbone(inputs, training=False)
        class_token = features[:, 0, :]  # (batch, 1024)

        # Scene classification head
        scene_features = tf.keras.layers.Dense(512, activation='relu')(class_token)
        scene_features = tf.keras.layers.Dropout(0.3)(scene_features)
        scene_logits = tf.keras.layers.Dense(num_scenes, name='scene')(scene_features)

        # Attribute classification head (multi-label)
        attr_features = tf.keras.layers.Dense(512, activation='relu')(class_token)
        attr_features = tf.keras.layers.Dropout(0.3)(attr_features)
        attr_logits = tf.keras.layers.Dense(num_attributes, name='attributes')(attr_features)

        # Aesthetic score regression head
        aesthetic_features = tf.keras.layers.Dense(256, activation='relu')(class_token)
        aesthetic_features = tf.keras.layers.Dropout(0.2)(aesthetic_features)
        aesthetic_score = tf.keras.layers.Dense(1, name='aesthetic')(aesthetic_features)

        self.model = tf.keras.Model(
            inputs=inputs,
            outputs={
                'scene': scene_logits,
                'attributes': attr_logits,
                'aesthetic': aesthetic_score
            }
        )

    def compile_model(self):
        """Compile with multi-task losses."""
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss={
                'scene': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                'attributes': tf.keras.losses.BinaryCrossentropy(from_logits=True),
                'aesthetic': 'mse'
            },
            loss_weights={
                'scene': 1.0,
                'attributes': 0.5,
                'aesthetic': 0.3
            },
            metrics={
                'scene': 'accuracy',
                'attributes': tf.keras.metrics.BinaryAccuracy(),
                'aesthetic': 'mae'
            }
        )

    def analyze_scene(self, image):
        """Comprehensive scene analysis."""
        if isinstance(image, str):
            img = tf.io.read_file(image)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [224, 224])
            img = img / 255.0
            img = tf.expand_dims(img, 0)
        else:
            img = image

        predictions = self.model(img, training=False)

        # Process outputs
        scene_probs = tf.nn.softmax(predictions['scene'])[0]
        top_scenes = tf.argsort(scene_probs, direction='DESCENDING')[:5]

        attr_probs = tf.nn.sigmoid(predictions['attributes'])[0]
        present_attrs = tf.where(attr_probs > 0.5)[:, 0]

        aesthetic = predictions['aesthetic'][0, 0].numpy()

        return {
            'top_scenes': [(int(idx), float(scene_probs[idx])) for idx in top_scenes],
            'attributes': [int(idx) for idx in present_attrs],
            'aesthetic_score': float(aesthetic),
            'scene_confidence': float(scene_probs[top_scenes[0]])
        }

# Usage
scene_analyzer = SceneUnderstanding(num_scenes=365, num_attributes=102)
scene_analyzer.compile_model()

# Train on multi-task dataset
# scene_analyzer.model.fit(train_data, epochs=30, ...)

# Analyze image
analysis = scene_analyzer.analyze_scene('photo.jpg')
print(f"Top scene: {analysis['top_scenes'][0]}")
print(f"Attributes detected: {len(analysis['attributes'])}")
print(f"Aesthetic score: {analysis['aesthetic_score']:.2f}/10")
```

#### Cross-Modal Retrieval (Image-Text)

```python
from vit_all_variants import create_vit_l32
import tensorflow as tf
import tensorflow_hub as hub

class CrossModalRetrieval:
    """Image-text retrieval using ViT-L/32 and text embeddings."""

    def __init__(self):
        # Image encoder: ViT-L/32
        self.image_encoder = create_vit_l32(image_size=224, include_top=False)

        # Text encoder: Universal Sentence Encoder (or similar)
        self.text_encoder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

        # Projection layers to shared embedding space
        self.image_projection = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(256)
        ])

        self.text_projection = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(256)
        ])

    def encode_images(self, images):
        """Encode images to shared embedding space."""
        features = self.image_encoder.extract_features(images)
        embeddings = self.image_projection(features)
        return tf.nn.l2_normalize(embeddings, axis=1)

    def encode_text(self, texts):
        """Encode text to shared embedding space."""
        features = self.text_encoder(texts)
        embeddings = self.text_projection(features)
        return tf.nn.l2_normalize(embeddings, axis=1)

    def train_with_contrastive_loss(self, image_text_pairs, epochs=20):
        """Train with contrastive loss to align modalities."""

        optimizer = tf.keras.optimizers.Adam(1e-4)
        temperature = 0.07

        for epoch in range(epochs):
            for images, texts in image_text_pairs:
                with tf.GradientTape() as tape:
                    # Encode both modalities
                    image_embeds = self.encode_images(images)
                    text_embeds = self.encode_text(texts)

                    # Compute similarity matrix
                    logits = tf.matmul(image_embeds, text_embeds, transpose_b=True) / temperature

                    # Contrastive loss (symmetric)
                    batch_size = tf.shape(images)[0]
                    labels = tf.range(batch_size)

                    loss_i2t = tf.keras.losses.sparse_categorical_crossentropy(
                        labels, logits, from_logits=True
                    )
                    loss_t2i = tf.keras.losses.sparse_categorical_crossentropy(
                        labels, tf.transpose(logits), from_logits=True
                    )

                    loss = (tf.reduce_mean(loss_i2t) + tf.reduce_mean(loss_t2i)) / 2

                # Update weights
                trainable_vars = (
                    self.image_encoder.trainable_variables +
                    self.text_projection.trainable_variables +
                    self.image_projection.trainable_variables
                )

                gradients = tape.gradient(loss, trainable_vars)
                optimizer.apply_gradients(zip(gradients, trainable_vars))

            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

    def search_by_text(self, query_text, image_database):
        """Search images using text query."""
        text_embed = self.encode_text([query_text])
        image_embeds = self.encode_images(image_database)

        # Compute similarities
        similarities = tf.matmul(text_embed, image_embeds, transpose_b=True)[0]
        top_k_indices = tf.argsort(similarities, direction='DESCENDING')

        return top_k_indices.numpy(), similarities.numpy()

    def search_by_image(self, query_image, text_database):
        """Search texts using image query."""
        image_embed = self.encode_images(tf.expand_dims(query_image, 0))
        text_embeds = self.encode_text(text_database)

        # Compute similarities
        similarities = tf.matmul(image_embed, text_embeds, transpose_b=True)[0]
        top_k_indices = tf.argsort(similarities, direction='DESCENDING')

        return top_k_indices.numpy(), similarities.numpy()

# Usage
retrieval = CrossModalRetrieval()

# Train on image-caption pairs
# retrieval.train_with_contrastive_loss(train_pairs, epochs=20)

# Text-to-image search
query = "a golden retriever playing in the park"
image_db = load_images()  # Your image database
top_images, scores = retrieval.search_by_text(query, image_db)

print(f"Top 5 images for '{query}':")
for i in range(5):
    print(f"{i+1}. Image {top_images[i]} (score: {scores[top_images[i]]:.3f})")

# Image-to-text search
query_img = load_image('query.jpg')
captions = ["dog playing", "cat sleeping", "beach sunset", ...]
top_captions, scores = retrieval.search_by_image(query_img, captions)

print(f"Top 3 matching captions:")
for i in range(3):
    print(f"{i+1}. '{captions[top_captions[i]]}' (score: {scores[top_captions[i]]:.3f})")
```

---

## ViT-H/14 (Huge, 14×14 patches)

### Model Description

ViT-H/14 is the largest and most powerful Vision Transformer variant, featuring 632 million parameters, 32 transformer layers, and 1280-dimensional embeddings. The unique 14×14 patch size maximizes spatial resolution with 257 tokens per 224×224 image, enabling exceptional fine-grained recognition. This model represents state-of-the-art performance (86-88% ImageNet accuracy) and is designed for research, flagship production systems, and applications demanding maximum accuracy.

**Key Specifications:**
- **Parameters**: ~632M
- **Layers**: 32 transformer blocks
- **Embedding Dimension**: 1280
- **Attention Heads**: 16
- **MLP Dimension**: 5120
- **Patch Size**: 14×14 pixels
- **Expected Accuracy**: ~86-88% on ImageNet-1K (with pre-training)

### Overview

**Best For:**
- State-of-the-art research applications
- Maximum accuracy requirements
- Fine-grained recognition at scale
- High-value production systems (autonomous vehicles, medical diagnostics)
- Benchmark comparisons and competitions
- Feature extraction for ultra-high-quality applications

**Advantages:**
- Highest accuracy of all ViT models
- Richest feature representations (1280-dim)
- Exceptional fine-grained recognition
- Maximum spatial resolution (257 tokens)
- Best transfer learning performance
- Superior out-of-distribution generalization

**Limitations:**
- Requires 40GB+ GPU memory for training (multi-GPU setup)
- Slowest inference (~195ms per image on V100)
- Highest deployment costs
- Significant energy consumption
- Risk of severe overfitting on small datasets
- 35+ days for pre-training from scratch

**Performance Characteristics:**
- Inference latency: ~195ms per image (V100 GPU), ~117ms (A100 GPU)
- Memory requirement: 16GB GPU RAM for inference, 40GB+ for training
- Training time: ~35 days for ImageNet-21K pre-training (2× GPU)
- FLOPs: 167.4G per forward pass
- Throughput: ~5 images/second (V100, batch size 4)
- Energy per inference: ~10 Wh

### Example Usage

#### State-of-the-Art Transfer Learning

```python
from vit_all_variants import create_vit_h14
import tensorflow as tf

# Load the huge model (requires significant GPU memory)
base_model = create_vit_h14(
    image_size=224,
    include_top=False,
    weights='path/to/imagenet21k_vith14_pretrained.h5'
)

# Strategy for multi-GPU training
strategy = tf.distribute.MirroredStrategy()

print(f"Number of GPUs: {strategy.num_replicas_in_sync}")

with strategy.scope():
    # Freeze backbone initially
    base_model.trainable = False

    # Build model
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    class_token = x[:, 0, :]  # (batch, 1280)

    # Deep classification head for maximum capacity
    x = tf.keras.layers.LayerNormalization()(class_token)
    x = tf.keras.layers.Dense(2560, activation='gelu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(1280, activation='gelu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(640, activation='gelu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compile with advanced optimizer
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=1e-3,
            weight_decay=0.05
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )

# Phase 1: Train head only (10 epochs)
print("Phase 1: Training classification head...")
history_head = model.fit(
    train_dataset.batch(16 * strategy.num_replicas_in_sync),
    validation_data=val_dataset.batch(16 * strategy.num_replicas_in_sync),
    epochs=10,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            'vit_h14_head.h5',
            save_best_only=True,
            monitor='val_accuracy'
        ),
        tf.keras.callbacks.TensorBoard(log_dir='logs/phase1')
    ]
)

# Phase 2: Fine-tune entire model with gradient checkpointing
print("Phase 2: Fine-tuning entire model...")

with strategy.scope():
    base_model.trainable = True

    # Very low learning rate for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=1e-6,
            weight_decay=0.05
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )

history_finetune = model.fit(
    train_dataset.batch(8 * strategy.num_replicas_in_sync),  # Smaller batch
    validation_data=val_dataset.batch(8 * strategy.num_replicas_in_sync),
    epochs=20,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            'vit_h14_finetuned_best.h5',
            save_best_only=True,
            monitor='val_accuracy'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.3,
            patience=3,
            min_lr=1e-8
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=7,
            restore_best_weights=True
        )
    ]
)

print(f"Final accuracy: {max(history_finetune.history['val_accuracy']):.4f}")
```

#### Medical Imaging: High-Stakes Classification

```python
from vit_all_variants import create_vit_h14
import tensorflow as tf
import numpy as np

class MedicalDiagnosticSystem:
    """High-accuracy diagnostic system using ViT-H/14."""

    def __init__(self, num_classes, image_size=384):
        """
        High-resolution medical imaging system.

        Args:
            num_classes: Number of diagnostic categories
            image_size: Higher resolution (384, 512, or 518)
        """
        self.image_size = image_size
        self.num_classes = num_classes

        # Use highest capacity model
        self.model = self._build_model()

    def _build_model(self):
        """Build model with uncertainty quantification."""

        base = create_vit_h14(
            image_size=self.image_size,
            include_top=False,
            weights='imagenet21k_pretrained.h5'
        )

        inputs = tf.keras.Input(shape=(self.image_size, self.image_size, 3))
        x = base(inputs)
        class_token = x[:, 0, :]

        # Multi-branch head for robust predictions
        # Branch 1: Primary classifier
        x1 = tf.keras.layers.Dense(2048, activation='gelu')(class_token)
        x1 = tf.keras.layers.Dropout(0.3)(x1)
        x1 = tf.keras.layers.Dense(1024, activation='gelu')(x1)
        x1 = tf.keras.layers.Dropout(0.2)(x1)
        logits1 = tf.keras.layers.Dense(self.num_classes, name='primary')(x1)

        # Branch 2: Auxiliary classifier (for regularization)
        x2 = tf.keras.layers.Dense(1024, activation='gelu')(class_token)
        x2 = tf.keras.layers.Dropout(0.3)(x2)
        logits2 = tf.keras.layers.Dense(self.num_classes, name='auxiliary')(x2)

        # Ensemble prediction
        logits = (logits1 + logits2) / 2

        model = tf.keras.Model(inputs=inputs, outputs=logits)
        return model

    def train(self, train_data, val_data, epochs=50):
        """Train with medical-grade protocols."""

        # Compute class weights for imbalanced medical datasets
        class_weights = self._compute_class_weights(train_data)

        self.model.compile(
            optimizer=tf.keras.optimizers.AdamW(1e-5, weight_decay=0.01),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc', multi_label=False),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')
            ]
        )

        # Advanced callbacks
        callbacks = [
            # Save best model
            tf.keras.callbacks.ModelCheckpoint(
                'medical_vit_h14_best.h5',
                monitor='val_auc',
                save_best_only=True,
                mode='max'
            ),

            # Early stopping with patience
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc',
                patience=15,
                restore_best_weights=True
            ),

            # Learning rate schedule
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-8
            ),

            # TensorBoard logging
            tf.keras.callbacks.TensorBoard(
                log_dir='logs/medical',
                histogram_freq=1
            )
        ]

        # Train
        history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            class_weight=class_weights,
            callbacks=callbacks
        )

        return history

    def predict_with_confidence(self, images, num_mc_samples=20):
        """
        Monte Carlo Dropout for uncertainty quantification.
        Critical for medical applications.
        """
        # Enable dropout at inference time
        predictions = []

        for _ in range(num_mc_samples):
            pred = self.model(images, training=True)
            pred_probs = tf.nn.softmax(pred)
            predictions.append(pred_probs)

        predictions = tf.stack(predictions)

        # Mean prediction
        mean_pred = tf.reduce_mean(predictions, axis=0)

        # Uncertainty metrics
        std_pred = tf.math.reduce_std(predictions, axis=0)

        # Predictive entropy (total uncertainty)
        entropy = -tf.reduce_sum(
            mean_pred * tf.math.log(mean_pred + 1e-10),
            axis=1
        )

        # Aleatoric uncertainty (data uncertainty)
        aleatoric = tf.reduce_mean(
            tf.reduce_sum(-predictions * tf.math.log(predictions + 1e-10), axis=2),
            axis=0
        )

        # Epistemic uncertainty (model uncertainty)
        epistemic = entropy - aleatoric

        return {
            'predictions': mean_pred.numpy(),
            'std': std_pred.numpy(),
            'entropy': entropy.numpy(),
            'epistemic_uncertainty': epistemic.numpy(),
            'aleatoric_uncertainty': aleatoric.numpy()
        }

    def clinical_report(self, image_path, diagnosis_names):
        """Generate clinical-grade diagnostic report."""

        # Load and preprocess
        img = tf.io.read_file(image_path)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.resize(img, [self.image_size, self.image_size])
        img = img / 255.0
        img = tf.expand_dims(img, 0)

        # Predict with uncertainty
        results = self.predict_with_confidence(img, num_mc_samples=30)

        pred = results['predictions'][0]
        top_class = pred.argmax()
        confidence = pred[top_class]
        uncertainty = results['entropy'][0]

        # Generate report
        report = {
            'primary_diagnosis': diagnosis_names[top_class],
            'confidence': float(confidence),
            'uncertainty': float(uncertainty),
            'recommendation': self._get_recommendation(confidence, uncertainty),
            'differential_diagnosis': []
        }

        # Add top 3 differential diagnoses
        top_3_indices = np.argsort(pred)[-3:][::-1]
        for idx in top_3_indices:
            report['differential_diagnosis'].append({
                'diagnosis': diagnosis_names[idx],
                'probability': float(pred[idx]),
                'std': float(results['std'][0][idx])
            })

        return report

    def _get_recommendation(self, confidence, uncertainty):
        """Clinical decision support."""
        if confidence > 0.9 and uncertainty < 0.3:
            return "High confidence diagnosis. Recommend standard treatment protocol."
        elif confidence > 0.7 and uncertainty < 0.5:
            return "Moderate confidence. Recommend correlation with clinical findings."
        else:
            return "⚠️ Low confidence or high uncertainty. MANDATORY specialist review required."

    def _compute_class_weights(self, train_data):
        """Compute balanced class weights."""
        # Extract labels from dataset
        labels = []
        for _, y in train_data:
            labels.extend(y.numpy())

        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight(
            'balanced',
            classes=np.arange(self.num_classes),
            y=labels
        )

        return dict(enumerate(class_weights))

# Usage
diagnostic_system = MedicalDiagnosticSystem(
    num_classes=10,  # e.g., chest X-ray pathologies
    image_size=512   # High resolution for medical imaging
)

# Train
history = diagnostic_system.train(
    train_dataset,
    val_dataset,
    epochs=100
)

# Clinical inference
diagnosis_names = ['Normal', 'Pneumonia', 'COVID-19', ...]
report = diagnostic_system.clinical_report('patient_xray.jpg', diagnosis_names)

print("DIAGNOSTIC REPORT")
print("="*50)
print(f"Primary Diagnosis: {report['primary_diagnosis']}")
print(f"Confidence: {report['confidence']:.1%}")
print(f"Uncertainty: {report['uncertainty']:.3f}")
print(f"\nRecommendation: {report['recommendation']}")
print(f"\nDifferential Diagnosis:")
for dd in report['differential_diagnosis']:
    print(f"  - {dd['diagnosis']}: {dd['probability']:.1%} ± {dd['std']:.2%}")
```

#### Vision-Language Model (CLIP-style)

```python
from vit_all_variants import create_vit_h14
import tensorflow as tf

class VisionLanguageModel:
    """CLIP-style model using ViT-H/14 for maximum quality."""

    def __init__(self, embed_dim=512):
        # Vision encoder: ViT-H/14
        self.vision_encoder = create_vit_h14(image_size=224, include_top=False)

        # Text encoder: Transformer
        self.text_encoder = self._build_text_encoder()

        # Projection to shared embedding space
        self.vision_projection = tf.keras.layers.Dense(embed_dim)
        self.text_projection = tf.keras.layers.Dense(embed_dim)

        # Temperature parameter (learnable)
        self.temperature = tf.Variable(np.log(1 / 0.07), trainable=True, dtype=tf.float32)

    def _build_text_encoder(self):
        """Build text transformer encoder."""
        # Simplified - in practice use BERT/RoBERTa
        return tf.keras.Sequential([
            tf.keras.layers.Embedding(50000, 512),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, return_sequences=True)),
            tf.keras.layers.GlobalAveragePooling1D()
        ])

    def encode_image(self, images):
        """Encode images to shared embedding space."""
        features = self.vision_encoder.extract_features(images)
        embeddings = self.vision_projection(features)
        return tf.nn.l2_normalize(embeddings, axis=1)

    def encode_text(self, texts):
        """Encode text to shared embedding space."""
        features = self.text_encoder(texts)
        embeddings = self.text_projection(features)
        return tf.nn.l2_normalize(embeddings, axis=1)

    def contrastive_loss(self, image_embeddings, text_embeddings):
        """CLIP-style contrastive loss."""
        # Compute similarity matrix
        temperature = tf.exp(self.temperature)
        logits = tf.matmul(image_embeddings, text_embeddings, transpose_b=True) * temperature

        # Labels: diagonal
        batch_size = tf.shape(logits)[0]
        labels = tf.range(batch_size)

        # Cross-entropy loss (symmetric)
        loss_i2t = tf.keras.losses.sparse_categorical_crossentropy(
            labels, logits, from_logits=True
        )
        loss_t2i = tf.keras.losses.sparse_categorical_crossentropy(
            labels, tf.transpose(logits), from_logits=True
        )

        return (tf.reduce_mean(loss_i2t) + tf.reduce_mean(loss_t2i)) / 2

    def train_step(self, images, texts):
        """Single training step."""
        with tf.GradientTape() as tape:
            image_embeds = self.encode_image(images)
            text_embeds = self.encode_text(texts)
            loss = self.contrastive_loss(image_embeds, text_embeds)

        # All trainable variables
        trainable_vars = (
            self.vision_encoder.trainable_variables +
            self.text_encoder.trainable_variables +
            self.vision_projection.trainable_variables +
            self.text_projection.trainable_variables +
            [self.temperature]
        )

        gradients = tape.gradient(loss, trainable_vars)
        return loss, gradients, trainable_vars

    def zero_shot_classify(self, images, class_descriptions):
        """Zero-shot classification using text descriptions."""
        # Encode image
        image_embeds = self.encode_image(images)

        # Encode all class descriptions
        text_embeds = self.encode_text(class_descriptions)

        # Compute similarities
        similarities = tf.matmul(image_embeds, text_embeds, transpose_b=True)
        similarities = similarities * tf.exp(self.temperature)

        # Softmax for probabilities
        probs = tf.nn.softmax(similarities, axis=1)

        return probs

# Usage
vlm = VisionLanguageModel(embed_dim=512)

# Training loop
optimizer = tf.keras.optimizers.Adam(1e-5)

for epoch in range(epochs):
    for images, captions in training_data:
        loss, gradients, variables = vlm.train_step(images, captions)
        optimizer.apply_gradients(zip(gradients, variables))

    print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Zero-shot classification
test_image = load_image('test.jpg')
class_descriptions = [
    "a photo of a dog",
    "a photo of a cat",
    "a photo of a bird",
    "a photo of a car"
]

probs = vlm.zero_shot_classify(tf.expand_dims(test_image, 0), class_descriptions)
predicted_class = tf.argmax(probs[0])

print(f"Predicted: {class_descriptions[predicted_class]}")
print(f"Confidence: {probs[0][predicted_class]:.1%}")
```

---

## ViT-H/16 (Huge, 16×16 patches)

### Model Description

ViT-H/16 offers the maximum model capacity (632M parameters, 32 layers, 1280-dim embeddings) with the standard 16×16 patch size, providing 197 tokens per image. This variant balances the representational power of the Huge architecture with slightly faster inference than ViT-H/14. It's ideal for applications requiring the highest quality features while maintaining compatibility with standard ViT configurations.

**Key Specifications:**
- **Parameters**: ~632M
- **Layers**: 32 transformer blocks
- **Embedding Dimension**: 1280
- **Attention Heads**: 16
- **MLP Dimension**: 5120
- **Patch Size**: 16×16 pixels
- **Expected Accuracy**: ~85-87% on ImageNet-1K (with pre-training)

### Overview

**Best For:**
- Maximum capacity feature extraction
- Research requiring highest-quality embeddings
- Transfer learning to highly diverse domains
- Ultra-high-accuracy production systems
- Benchmark comparisons
- Foundation models for downstream tasks

**Advantages:**
- Highest capacity (632M parameters)
- Richest 1280-dimensional features
- Faster than ViT-H/14 (fewer tokens)
- Excellent transfer learning
- Strong performance across diverse tasks
- Better speed-accuracy tradeoff than H/14

**Limitations:**
- Still requires 40GB+ GPU for training
- High inference cost (~140ms per image)
- Significant deployment costs
- Long training times
- Risk of overfitting on small datasets

**Performance Characteristics:**
- Inference latency: ~140ms per image (V100 GPU), ~84ms (A100 GPU)
- Memory requirement: 14GB GPU RAM for inference, 40GB+ for training
- Training time: ~30 days for ImageNet-21K pre-training (2× GPU)
- FLOPs: 124.6G per forward pass
- Throughput: ~7 images/second (V100, batch size 4)

### Example Usage

#### Maximum-Quality Feature Extraction

```python
from vit_all_variants import create_vit_h16
import tensorflow as tf
import numpy as np

class UniversalFeatureExtractor:
    """Ultra-high-quality feature extraction using ViT-H/16."""

    def __init__(self, weights_path=None):
        self.model = create_vit_h16(
            image_size=224,
            include_top=False,
            weights=weights_path
        )

    def extract_features(self, images, output_type='class_token'):
        """
        Extract features from images.

        Args:
            images: Input images (batch, 224, 224, 3)
            output_type: 'class_token', 'all_tokens', or 'spatial_avg'

        Returns:
            Features of shape:
            - class_token: (batch, 1280)
            - all_tokens: (batch, 197, 1280)
            - spatial_avg: (batch, 1280)
        """
        all_tokens = self.model(images, training=False)

        if output_type == 'class_token':
            return all_tokens[:, 0, :]
        elif output_type == 'all_tokens':
            return all_tokens
        elif output_type == 'spatial_avg':
            # Average over all patch tokens (excluding class token)
            return tf.reduce_mean(all_tokens[:, 1:, :], axis=1)
        else:
            raise ValueError(f"Unknown output_type: {output_type}")

    def extract_multiscale_features(self, images, scales=[224, 384, 512]):
        """Extract features at multiple scales and concatenate."""
        all_features = []

        for scale in scales:
            # Resize to scale
            scaled_images = tf.image.resize(images, [scale, scale])

            # Extract features
            features = self.extract_features(scaled_images, output_type='class_token')
            all_features.append(features)

        # Concatenate multi-scale features
        combined = tf.concat(all_features, axis=1)
        return combined

    def compute_image_embeddings_batch(self, image_paths, batch_size=8):
        """Efficiently compute embeddings for large image collections."""
        embeddings = []

        dataset = tf.data.Dataset.from_tensor_slices(image_paths)

        def load_and_preprocess(path):
            img = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [224, 224])
            return img / 255.0

        dataset = dataset.map(load_and_preprocess).batch(batch_size)

        for batch in dataset:
            batch_embeddings = self.extract_features(batch)
            embeddings.append(batch_embeddings.numpy())

        return np.vstack(embeddings)

# Usage
extractor = UniversalFeatureExtractor(weights_path='vit_h16_pretrained.h5')

# Single image features
image = load_image('photo.jpg')
features = extractor.extract_features(tf.expand_dims(image, 0))
print(f"Feature shape: {features.shape}")  # (1, 1280)

# Multi-scale features
multi_scale = extractor.extract_multiscale_features(
    tf.expand_dims(image, 0),
    scales=[224, 384, 512]
)
print(f"Multi-scale features: {multi_scale.shape}")  # (1, 3840)

# Batch extraction
image_list = ['img1.jpg', 'img2.jpg', ...]
embeddings = extractor.compute_image_embeddings_batch(image_list, batch_size=4)
print(f"Extracted {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
```

#### Few-Shot Learning

```python
from vit_all_variants import create_vit_h16
import tensorflow as tf
import numpy as np

class FewShotClassifier:
    """Few-shot learning using ViT-H/16 features."""

    def __init__(self):
        self.feature_extractor = create_vit_h16(
            image_size=224,
            include_top=False,
            weights='imagenet21k_pretrained.h5'
        )
        # Freeze feature extractor
        self.feature_extractor.trainable = False

    def extract_features(self, images):
        """Extract normalized features."""
        features = self.feature_extractor.extract_features(images)
        return tf.nn.l2_normalize(features, axis=1)

    def build_prototypes(self, support_images, support_labels, n_way):
        """
        Build class prototypes from support set.

        Args:
            support_images: Support set images (n_way * k_shot, 224, 224, 3)
            support_labels: Support set labels (n_way * k_shot,)
            n_way: Number of classes

        Returns:
            Class prototypes: (n_way, 1280)
        """
        # Extract features
        support_features = self.extract_features(support_images)

        # Compute prototypes (mean of each class)
        prototypes = []
        for class_id in range(n_way):
            class_mask = support_labels == class_id
            class_features = tf.boolean_mask(support_features, class_mask)
            prototype = tf.reduce_mean(class_features, axis=0)
            prototypes.append(prototype)

        return tf.stack(prototypes)

    def classify_query(self, query_images, prototypes):
        """
        Classify query images using prototypical networks.

        Args:
            query_images: Query images (batch, 224, 224, 3)
            prototypes: Class prototypes (n_way, 1280)

        Returns:
            Predicted class probabilities
        """
        # Extract query features
        query_features = self.extract_features(query_images)

        # Compute distances to prototypes
        distances = tf.norm(
            tf.expand_dims(query_features, 1) - tf.expand_dims(prototypes, 0),
            axis=2
        )

        # Convert distances to probabilities (negative distance)
        logits = -distances
        probs = tf.nn.softmax(logits, axis=1)

        return probs

    def evaluate_few_shot(self, support_set, query_set, n_way=5, k_shot=5):
        """
        Evaluate few-shot classification.

        Args:
            support_set: (images, labels) for support
            query_set: (images, labels) for query
            n_way: Number of classes
            k_shot: Number of examples per class
        """
        support_images, support_labels = support_set
        query_images, query_labels = query_set

        # Build prototypes
        prototypes = self.build_prototypes(support_images, support_labels, n_way)

        # Classify queries
        predictions = self.classify_query(query_images, prototypes)
        predicted_classes = tf.argmax(predictions, axis=1)

        # Compute accuracy
        accuracy = tf.reduce_mean(
            tf.cast(predicted_classes == query_labels, tf.float32)
        )

        return accuracy.numpy(), predictions.numpy()

# Usage: 5-way 5-shot classification
few_shot = FewShotClassifier()

# Prepare support set (5 classes, 5 examples each)
support_images = load_support_images()  # (25, 224, 224, 3)
support_labels = np.repeat(np.arange(5), 5)  # [0,0,0,0,0, 1,1,1,1,1, ...]

# Prepare query set
query_images = load_query_images()  # (50, 224, 224, 3)
query_labels = load_query_labels()  # (50,)

# Evaluate
accuracy, predictions = few_shot.evaluate_few_shot(
    support_set=(support_images, support_labels),
    query_set=(query_images, query_labels),
    n_way=5,
    k_shot=5
)

print(f"5-way 5-shot accuracy: {accuracy:.2%}")
```

#### Domain Adaptation

```python
from vit_all_variants import create_vit_h16
import tensorflow as tf

class DomainAdaptationModel:
    """Domain adaptation using ViT-H/16 with adversarial training."""

    def __init__(self, num_classes):
        # Shared feature extractor
        self.feature_extractor = create_vit_h16(
            image_size=224,
            include_top=False,
            weights='imagenet21k_pretrained.h5'
        )

        # Task classifier
        self.task_classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_classes)
        ])

        # Domain discriminator
        self.domain_discriminator = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    @tf.function
    def train_step(self, source_images, source_labels, target_images, lambda_adv=0.1):
        """
        Single training step with adversarial domain adaptation.

        Args:
            source_images: Source domain images (labeled)
            source_labels: Source domain labels
            target_images: Target domain images (unlabeled)
            lambda_adv: Weight for adversarial loss
        """
        with tf.GradientTape(persistent=True) as tape:
            # Extract features
            source_features = self.feature_extractor.extract_features(source_images)
            target_features = self.feature_extractor.extract_features(target_images)

            # Task classification (source domain)
            source_logits = self.task_classifier(source_features, training=True)
            task_loss = tf.keras.losses.sparse_categorical_crossentropy(
                source_labels,
                source_logits,
                from_logits=True
            )
            task_loss = tf.reduce_mean(task_loss)

            # Domain classification
            source_domain_pred = self.domain_discriminator(source_features, training=True)
            target_domain_pred = self.domain_discriminator(target_features, training=True)

            # Domain discriminator loss (0=source, 1=target)
            domain_loss_source = tf.keras.losses.binary_crossentropy(
                tf.zeros_like(source_domain_pred),
                source_domain_pred
            )
            domain_loss_target = tf.keras.losses.binary_crossentropy(
                tf.ones_like(target_domain_pred),
                target_domain_pred
            )
            domain_loss = tf.reduce_mean(domain_loss_source + domain_loss_target)

            # Feature extractor loss (confuse discriminator)
            confusion_loss = -domain_loss

            # Total loss
            total_loss = task_loss + lambda_adv * confusion_loss

        # Update feature extractor and task classifier
        fe_tc_vars = (
            self.feature_extractor.trainable_variables +
            self.task_classifier.trainable_variables
        )
        fe_tc_grads = tape.gradient(total_loss, fe_tc_vars)

        # Update domain discriminator
        disc_vars = self.domain_discriminator.trainable_variables
        disc_grads = tape.gradient(domain_loss, disc_vars)

        del tape

        return task_loss, domain_loss, fe_tc_grads, disc_grads, fe_tc_vars, disc_vars

    def predict(self, images):
        """Predict on target domain."""
        features = self.feature_extractor.extract_features(images)
        logits = self.task_classifier(features, training=False)
        return tf.nn.softmax(logits)

# Usage
adapter = DomainAdaptationModel(num_classes=10)

# Optimizers
optimizer_main = tf.keras.optimizers.Adam(1e-5)
optimizer_disc = tf.keras.optimizers.Adam(1e-4)

# Training loop
for epoch in range(epochs):
    for (source_images, source_labels), target_images in zip(source_data, target_data):
        task_loss, domain_loss, main_grads, disc_grads, main_vars, disc_vars = \
            adapter.train_step(source_images, source_labels, target_images, lambda_adv=0.1)

        optimizer_main.apply_gradients(zip(main_grads, main_vars))
        optimizer_disc.apply_gradients(zip(disc_grads, disc_vars))

    print(f"Epoch {epoch}: Task Loss={task_loss:.4f}, Domain Loss={domain_loss:.4f}")

# Evaluate on target domain
target_predictions = adapter.predict(target_test_images)
target_accuracy = compute_accuracy(target_predictions, target_test_labels)
print(f"Target domain accuracy: {target_accuracy:.2%}")
```

#### Model Compression and Distillation

```python
from vit_all_variants import create_vit_h16, create_vit_b16
import tensorflow as tf

class ModelDistiller:
    """Distill ViT-H/16 knowledge into smaller model."""

    def __init__(self, temperature=4.0, alpha=0.7):
        """
        Args:
            temperature: Softmax temperature for distillation
            alpha: Weight for distillation loss (1-alpha for hard labels)
        """
        # Teacher: ViT-H/16
        self.teacher = create_vit_h16(
            image_size=224,
            include_top=True,
            num_classes=1000,
            weights='vit_h16_imagenet_trained.h5'
        )
        self.teacher.trainable = False

        # Student: ViT-B/16 (4× smaller)
        self.student = create_vit_b16(
            image_size=224,
            include_top=True,
            num_classes=1000
        )

        self.temperature = temperature
        self.alpha = alpha

    def distillation_loss(self, student_logits, teacher_logits, hard_labels):
        """
        Combined distillation loss.

        Args:
            student_logits: Student model predictions (logits)
            teacher_logits: Teacher model predictions (logits)
            hard_labels: Ground truth labels

        Returns:
            Combined loss
        """
        # Soft targets from teacher
        teacher_probs = tf.nn.softmax(teacher_logits / self.temperature)
        student_probs = tf.nn.softmax(student_logits / self.temperature)

        # Distillation loss (KL divergence)
        distill_loss = tf.keras.losses.categorical_crossentropy(
            teacher_probs,
            student_probs
        )
        distill_loss = tf.reduce_mean(distill_loss) * (self.temperature ** 2)

        # Hard label loss
        hard_loss = tf.keras.losses.sparse_categorical_crossentropy(
            hard_labels,
            student_logits,
            from_logits=True
        )
        hard_loss = tf.reduce_mean(hard_loss)

        # Combined loss
        total_loss = self.alpha * distill_loss + (1 - self.alpha) * hard_loss

        return total_loss, distill_loss, hard_loss

    @tf.function
    def train_step(self, images, labels):
        """Single distillation training step."""
        with tf.GradientTape() as tape:
            # Get teacher predictions (no gradient)
            teacher_logits = self.teacher(images, training=False)

            # Get student predictions
            student_logits = self.student(images, training=True)

            # Compute loss
            total_loss, distill_loss, hard_loss = self.distillation_loss(
                student_logits,
                teacher_logits,
                labels
            )

        # Update student
        gradients = tape.gradient(total_loss, self.student.trainable_variables)

        return total_loss, distill_loss, hard_loss, gradients

    def distill(self, train_dataset, val_dataset, epochs=30):
        """Distill teacher knowledge to student."""

        optimizer = tf.keras.optimizers.Adam(1e-4)

        best_val_acc = 0.0

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")

            # Training
            train_loss = []
            for images, labels in train_dataset:
                total_loss, distill_loss, hard_loss, grads = self.train_step(images, labels)
                optimizer.apply_gradients(zip(grads, self.student.trainable_variables))
                train_loss.append(total_loss.numpy())

            avg_train_loss = np.mean(train_loss)

            # Validation
            val_acc = self.evaluate(val_dataset)

            print(f"Train Loss: {avg_train_loss:.4f}, Val Accuracy: {val_acc:.2%}")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.student.save_weights('distilled_vit_b16_best.h5')
                print(f"✓ New best model saved (accuracy: {val_acc:.2%})")

        print(f"\nDistillation complete! Best accuracy: {best_val_acc:.2%}")
        print(f"Teacher params: 632M → Student params: 86M (7.4× reduction)")

    def evaluate(self, dataset):
        """Evaluate student model."""
        correct = 0
        total = 0

        for images, labels in dataset:
            predictions = self.student(images, training=False)
            predicted_classes = tf.argmax(predictions, axis=1)
            correct += tf.reduce_sum(tf.cast(predicted_classes == labels, tf.int32))
            total += len(labels)

        return float(correct) / float(total)

# Usage
distiller = ModelDistiller(temperature=4.0, alpha=0.7)

# Distill on ImageNet or your dataset
distiller.distill(
    train_dataset=train_ds,
    val_dataset=val_ds,
    epochs=30
)

# The distilled student model is now 7.4× smaller but retains
# much of the teacher's performance!
```

---

## Summary Table

Quick reference for choosing the right model:

| Model    | Params | Speed     | Accuracy | Best Use Case                          |
|----------|--------|-----------|----------|----------------------------------------|
| ViT-S/32 | 22M    | Very Fast | 75%      | Mobile, real-time, prototyping         |
| ViT-S/16 | 22M    | Fast      | 78%      | Edge devices, small datasets           |
| ViT-B/32 | 86M    | Moderate  | 75-77%   | Balanced performance, cloud deployment |
| ViT-B/16 | 86M    | Moderate  | 80-82%   | General purpose, transfer learning     |
| ViT-L/32 | 307M   | Slow      | 77-79%   | Large-scale retrieval, rich features   |
| ViT-L/16 | 307M   | Slow      | 83-85%   | High accuracy, research                |
| ViT-H/14 | 632M   | Very Slow | 86-88%   | State-of-the-art, fine-grained tasks   |
| ViT-H/16 | 632M   | Very Slow | 85-87%   | Maximum capacity, feature extraction   |

**Need help choosing?**
- Start with **ViT-B/16** for most applications
- Use **ViT-S/32** for speed-critical applications
- Choose **ViT-L/16** or **ViT-H/14** when accuracy is paramount
- Select */32 variants for faster inference with acceptable accuracy trade-off
