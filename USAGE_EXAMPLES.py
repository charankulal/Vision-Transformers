"""
Vision Transformer (ViT) - All Variants Usage Examples

This file demonstrates how to use all 8 ViT model variations:
- ViT-S/16, ViT-S/32 (Small models)
- ViT-B/16, ViT-B/32 (Base models)
- ViT-L/16, ViT-L/32 (Large models)
- ViT-H/14, ViT-H/16 (Huge models)

Author: Vision Transformer Implementation
"""

import tensorflow as tf
import numpy as np
from vit_all_variants import (
    create_vit_s16,
    create_vit_s32,
    create_vit_b16,
    create_vit_b32,
    create_vit_l16,
    create_vit_l32,
    create_vit_h14,
    create_vit_h16,
    create_vit,  # Generic factory function
    list_variants,
    compare_variants
)


# =============================================================================
# EXAMPLE 1: Creating Different Model Variants
# =============================================================================

def example_1_create_models():
    """Demonstrate how to create each model variant."""
    print("="*80)
    print("EXAMPLE 1: Creating Different Model Variants")
    print("="*80)

    # Method 1: Using specific factory functions
    print("\n[Method 1] Using specific factory functions:")

    vit_s16 = create_vit_s16(image_size=224, include_top=False)
    print(f"✓ ViT-S/16 created - Output shape: (batch, 197, 384)")

    vit_s32 = create_vit_s32(image_size=224, include_top=False)
    print(f"✓ ViT-S/32 created - Output shape: (batch, 50, 384)")

    vit_b16 = create_vit_b16(image_size=224, include_top=False)
    print(f"✓ ViT-B/16 created - Output shape: (batch, 197, 768)")

    vit_b32 = create_vit_b32(image_size=224, include_top=False)
    print(f"✓ ViT-B/32 created - Output shape: (batch, 50, 768)")

    vit_l16 = create_vit_l16(image_size=224, include_top=False)
    print(f"✓ ViT-L/16 created - Output shape: (batch, 197, 1024)")

    vit_l32 = create_vit_l32(image_size=224, include_top=False)
    print(f"✓ ViT-L/32 created - Output shape: (batch, 50, 1024)")

    vit_h14 = create_vit_h14(image_size=224, include_top=False)
    print(f"✓ ViT-H/14 created - Output shape: (batch, 257, 1280)")

    vit_h16 = create_vit_h16(image_size=224, include_top=False)
    print(f"✓ ViT-H/16 created - Output shape: (batch, 197, 1280)")

    # Method 2: Using generic factory function
    print("\n[Method 2] Using generic factory function:")

    model = create_vit(variant="vit_b16", image_size=224, include_top=False)
    print(f"✓ Created ViT-B/16 using create_vit()")

    # Method 3: List all available variants
    print("\n[Method 3] List all available variants:")
    list_variants()


# =============================================================================
# EXAMPLE 2: Basic Feature Extraction
# =============================================================================

def example_2_feature_extraction():
    """Extract features using different model variants."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Feature Extraction with Different Variants")
    print("="*80)

    # Create sample images
    batch_size = 4
    images = tf.random.normal((batch_size, 224, 224, 3))
    print(f"\nInput images shape: {images.shape}")

    # Compare feature extraction across variants
    variants = {
        'ViT-S/32': create_vit_s32(image_size=224, include_top=False),
        'ViT-S/16': create_vit_s16(image_size=224, include_top=False),
        'ViT-B/32': create_vit_b32(image_size=224, include_top=False),
        'ViT-B/16': create_vit_b16(image_size=224, include_top=False),
    }

    print("\nExtracting features from each variant:")
    print("-" * 80)

    for name, model in variants.items():
        # Extract all tokens
        all_tokens = model(images, training=False)

        # Extract class token only
        class_token = model.extract_features(images, training=False)

        print(f"{name:12} | All tokens: {all_tokens.shape} | Class token: {class_token.shape}")

    # Detailed example with ViT-B/16
    print("\n[Detailed Example] Using ViT-B/16:")
    model = create_vit_b16(image_size=224, include_top=False)

    features = model(images, training=False)
    print(f"  - Full output shape: {features.shape}")
    print(f"  - Number of tokens: {features.shape[1]} (1 class token + {features.shape[1]-1} patch tokens)")
    print(f"  - Embedding dimension: {features.shape[2]}")

    # Extract class token for classification/similarity tasks
    class_features = model.extract_features(images, training=False)
    print(f"  - Class token shape: {class_features.shape}")
    print(f"  - Use class token for: classification, similarity search, clustering")


# =============================================================================
# EXAMPLE 3: Image Classification
# =============================================================================

def example_3_classification():
    """Use models for image classification."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Image Classification")
    print("="*80)

    num_classes = 10  # e.g., CIFAR-10

    # Create models with classification heads
    print("\n[Creating classification models]")

    models = {
        'ViT-S/32': create_vit_s32(image_size=224, include_top=True, num_classes=num_classes),
        'ViT-B/16': create_vit_b16(image_size=224, include_top=True, num_classes=num_classes),
    }

    # Sample input
    images = tf.random.normal((8, 224, 224, 3))

    for name, model in models.items():
        print(f"\n{name}:")

        # Get predictions (logits)
        logits = model(images, training=False)
        print(f"  - Logits shape: {logits.shape}")

        # Convert to probabilities
        probabilities = tf.nn.softmax(logits)
        predicted_classes = tf.argmax(probabilities, axis=1)

        print(f"  - Predicted classes: {predicted_classes.numpy()}")
        print(f"  - Top confidence scores: {tf.reduce_max(probabilities, axis=1).numpy()}")

    # Compile and train example
    print("\n[Example: Compile and train]")
    model = create_vit_b16(image_size=224, include_top=True, num_classes=num_classes, dropout=0.1)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    print("✓ Model compiled and ready for training")
    print("  Use: model.fit(train_dataset, validation_data=val_dataset, epochs=10)")


# =============================================================================
# EXAMPLE 4: Transfer Learning
# =============================================================================

def example_4_transfer_learning():
    """Demonstrate transfer learning with frozen backbone."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Transfer Learning with Frozen Backbone")
    print("="*80)

    num_classes = 5

    # Create base model and freeze it
    print("\n[Step 1] Create and freeze base model:")
    base_model = create_vit_b16(image_size=224, include_top=False)
    base_model.trainable = False
    print(f"✓ Base model created and frozen")
    print(f"  - Trainable parameters: {sum([tf.size(v).numpy() for v in base_model.trainable_variables]):,}")

    # Build custom model
    print("\n[Step 2] Build custom classification head:")
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)

    # Extract class token
    x = x[:, 0, :]  # Shape: (batch, 768)
    print(f"  - Class token shape: (batch, 768)")

    # Add custom layers
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    print(f"✓ Custom model built")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Count parameters
    total_params = sum([tf.size(v).numpy() for v in model.trainable_variables])
    print(f"  - Trainable parameters: {total_params:,} (only the custom head)")

    # Test forward pass
    print("\n[Step 3] Test forward pass:")
    test_images = tf.random.normal((4, 224, 224, 3))
    predictions = model(test_images, training=False)
    print(f"  - Input shape: {test_images.shape}")
    print(f"  - Output shape: {predictions.shape}")
    print(f"✓ Model ready for training on your dataset!")


# =============================================================================
# EXAMPLE 5: Using Different Image Sizes
# =============================================================================

def example_5_different_image_sizes():
    """Demonstrate using different image sizes."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Using Different Image Sizes")
    print("="*80)

    print("\nViT models work with any image size divisible by the patch size:")

    # ViT-B/16: Image size must be divisible by 16
    print("\n[ViT-B/16] Image sizes divisible by 16:")
    for img_size in [224, 256, 384, 448, 512]:
        model = create_vit_b16(image_size=img_size, include_top=False)

        # Calculate number of patches
        num_patches = (img_size // 16) ** 2
        total_tokens = num_patches + 1  # +1 for class token

        # Test with sample image
        test_img = tf.random.normal((1, img_size, img_size, 3))
        output = model(test_img, training=False)

        print(f"  Image {img_size}×{img_size}: {num_patches} patches, {total_tokens} tokens → {output.shape}")

    # ViT-B/32: Image size must be divisible by 32
    print("\n[ViT-B/32] Image sizes divisible by 32:")
    for img_size in [224, 256, 320, 384, 512]:
        model = create_vit_b32(image_size=img_size, include_top=False)

        num_patches = (img_size // 32) ** 2
        total_tokens = num_patches + 1

        test_img = tf.random.normal((1, img_size, img_size, 3))
        output = model(test_img, training=False)

        print(f"  Image {img_size}×{img_size}: {num_patches} patches, {total_tokens} tokens → {output.shape}")

    # Higher resolution for fine-grained tasks
    print("\n[Recommendation] Higher resolution for fine-grained recognition:")
    print("  - Standard tasks: 224×224")
    print("  - Fine-grained recognition: 384×384 or 448×448")
    print("  - Medical imaging: 512×512 or higher")


# =============================================================================
# EXAMPLE 6: Comparing Model Variants
# =============================================================================

def example_6_compare_variants():
    """Compare all model variants side by side."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Comparing Model Variants")
    print("="*80)

    # Use built-in comparison function
    print("\n[Built-in Comparison Function]")
    compare_variants(image_size=224)

    # Manual comparison with timing
    print("\n[Speed Comparison] Inference time (single forward pass):")
    print("-" * 80)

    variants = {
        'ViT-S/32': create_vit_s32(image_size=224, include_top=False),
        'ViT-S/16': create_vit_s16(image_size=224, include_top=False),
        'ViT-B/32': create_vit_b32(image_size=224, include_top=False),
        'ViT-B/16': create_vit_b16(image_size=224, include_top=False),
    }

    # Warmup and timing
    test_image = tf.random.normal((1, 224, 224, 3))

    for name, model in variants.items():
        # Warmup
        for _ in range(5):
            _ = model(test_image, training=False)

        # Time it
        import time
        start = time.time()
        iterations = 20
        for _ in range(iterations):
            _ = model(test_image, training=False)
        end = time.time()

        avg_time_ms = (end - start) / iterations * 1000
        print(f"{name:12} | {avg_time_ms:.2f} ms per image")

    print("\nNote: Actual inference time depends on hardware (CPU/GPU)")


# =============================================================================
# EXAMPLE 7: Saving and Loading Models
# =============================================================================

def example_7_save_load_models():
    """Demonstrate saving and loading models."""
    print("\n" + "="*80)
    print("EXAMPLE 7: Saving and Loading Models")
    print("="*80)

    # Create a model
    print("\n[Step 1] Create and configure model:")
    model = create_vit_b16(
        image_size=224,
        include_top=True,
        num_classes=10,
        dropout=0.1
    )
    print("✓ Model created")

    # Build model (required before saving)
    model.build((None, 224, 224, 3))
    print("✓ Model built")

    # Method 1: Save weights only
    print("\n[Method 1] Save weights only:")
    weights_path = "vit_b16_weights.h5"
    model.save_weights(weights_path)
    print(f"✓ Weights saved to: {weights_path}")

    # Load weights
    new_model = create_vit_b16(
        image_size=224,
        include_top=True,
        num_classes=10,
        dropout=0.1
    )
    new_model.build((None, 224, 224, 3))
    new_model.load_weights(weights_path)
    print(f"✓ Weights loaded from: {weights_path}")

    # Method 2: Save entire model (TensorFlow SavedModel format)
    print("\n[Method 2] Save entire model (recommended for deployment):")
    model_path = "vit_b16_model"
    tf.saved_model.save(model, model_path)
    print(f"✓ Model saved to: {model_path}/")

    # Load entire model
    loaded_model = tf.saved_model.load(model_path)
    print(f"✓ Model loaded from: {model_path}/")

    # Method 3: Using weights parameter during creation
    print("\n[Method 3] Load weights during model creation:")
    model_with_weights = create_vit_b16(
        image_size=224,
        include_top=True,
        num_classes=10,
        weights=weights_path
    )
    print(f"✓ Model created with pre-loaded weights")

    print("\n[Best Practices]")
    print("  - For training checkpoints: Use save_weights() / load_weights()")
    print("  - For deployment: Use tf.saved_model.save()")
    print("  - For transfer learning: Load pre-trained ImageNet weights if available")


# =============================================================================
# EXAMPLE 8: Batch Processing for Large Datasets
# =============================================================================

def example_8_batch_processing():
    """Efficiently process large image collections."""
    print("\n" + "="*80)
    print("EXAMPLE 8: Batch Processing for Large Datasets")
    print("="*80)

    # Create model
    model = create_vit_b16(image_size=224, include_top=False)
    print("✓ Model created: ViT-B/16")

    # Simulate large dataset
    print("\n[Scenario] Extract features from 1000 images efficiently")

    num_images = 1000
    batch_size = 32

    # Create dummy dataset
    def generate_dummy_images():
        for i in range(num_images):
            # Simulate loading an image
            img = tf.random.normal((224, 224, 3))
            yield img

    # Build efficient pipeline
    print(f"\n[Building data pipeline with batch_size={batch_size}]")

    dataset = tf.data.Dataset.from_generator(
        generate_dummy_images,
        output_signature=tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32)
    )

    # Optimize pipeline
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    print("✓ Data pipeline created with optimizations:")
    print(f"  - Batching: {batch_size} images per batch")
    print(f"  - Prefetching: Automatic (tf.data.AUTOTUNE)")

    # Process batches
    print(f"\n[Processing {num_images} images...]")

    all_features = []
    num_batches = 0

    for batch in dataset:
        # Extract features
        features = model.extract_features(batch, training=False)
        all_features.append(features.numpy())
        num_batches += 1

        if num_batches % 10 == 0:
            print(f"  Processed {num_batches * batch_size}/{num_images} images...")

    # Concatenate all features
    all_features = np.vstack(all_features)

    print(f"\n✓ Processing complete!")
    print(f"  - Total features extracted: {all_features.shape}")
    print(f"  - Shape: ({all_features.shape[0]} images, {all_features.shape[1]} dimensions)")
    print(f"  - Memory usage: ~{all_features.nbytes / 1024 / 1024:.2f} MB")


# =============================================================================
# EXAMPLE 9: Choosing the Right Model
# =============================================================================

def example_9_model_selection_guide():
    """Guide for choosing the right model variant."""
    print("\n" + "="*80)
    print("EXAMPLE 9: Model Selection Guide")
    print("="*80)

    guide = """
    🎯 QUICK SELECTION GUIDE
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    📱 MOBILE & EDGE DEPLOYMENT
       → Use: ViT-S/32 or ViT-S/32
       → Why: Smallest models, fastest inference, can run on mobile devices
       → Example: Real-time apps, embedded systems, mobile classifiers

    ⚡ NEED SPEED (Cloud/Server)
       → Use: ViT-S/32 or ViT-B/32
       → Why: Fewer tokens (32×32 patches) = faster processing
       → Example: Real-time video processing, high-throughput batch jobs

    🎯 BALANCED PERFORMANCE (Most Common)
       → Use: ViT-B/16
       → Why: Industry standard, best accuracy/speed trade-off
       → Example: General classification, transfer learning, Kaggle competitions

    🏆 MAXIMUM ACCURACY
       → Use: ViT-L/16 or ViT-H/14
       → Why: Highest capacity models, state-of-the-art performance
       → Example: Medical imaging, fine-grained recognition, research

    🔬 RESEARCH & EXPERIMENTS
       → Use: ViT-B/16 (start) → ViT-L/16 (if needed)
       → Why: Well-studied baseline, extensive literature
       → Example: Academic papers, benchmark comparisons

    💰 COST-EFFICIENT CLOUD
       → Use: ViT-B/32
       → Why: Good accuracy with moderate compute costs
       → Example: Cloud APIs, cost-sensitive production systems

    🔍 FINE-GRAINED RECOGNITION
       → Use: ViT-H/14 (best) or ViT-L/16 (good)
       → Why: More spatial tokens capture fine details
       → Example: Bird species, medical pathology, satellite imagery

    📊 FEATURE EXTRACTION
       → Use: ViT-B/16 or ViT-L/16
       → Why: Rich embeddings (768 or 1024 dim) for downstream tasks
       → Example: Similarity search, clustering, retrieval systems

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    💡 GENERAL RULES:
       • Patch size /32 = Faster but slightly lower accuracy
       • Patch size /16 = Slower but higher accuracy
       • Patch size /14 = Maximum tokens, best for fine details
       • Start with ViT-B/16, scale up/down based on needs
       • Use smaller models (S variants) for small datasets to avoid overfitting
    """

    print(guide)

    # Interactive recommendation
    print("\n[INTERACTIVE RECOMMENDATION]")
    print("Based on common scenarios:\n")

    scenarios = {
        "Kaggle Competition": "ViT-B/16",
        "Mobile App": "ViT-S/32",
        "Medical Diagnosis": "ViT-H/14 or ViT-L/16",
        "Real-time Video": "ViT-S/32 or ViT-B/32",
        "Image Retrieval": "ViT-B/16",
        "Fine-grained Birds": "ViT-L/16 or ViT-H/14",
        "Content Moderation": "ViT-B/32",
        "Research Baseline": "ViT-B/16",
    }

    for scenario, recommendation in scenarios.items():
        print(f"  {scenario:25} → {recommendation}")


# =============================================================================
# EXAMPLE 10: Complete End-to-End Workflow
# =============================================================================

def example_10_complete_workflow():
    """Complete workflow from data to trained model."""
    print("\n" + "="*80)
    print("EXAMPLE 10: Complete End-to-End Workflow")
    print("="*80)

    workflow = """
    📋 COMPLETE WORKFLOW EXAMPLE
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    1️⃣  CHOOSE MODEL VARIANT
        model = create_vit_b16(image_size=224, include_top=True,
                               num_classes=10, dropout=0.1)

    2️⃣  PREPARE DATA
        # Load images and create dataset
        train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        train_ds = train_ds.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)

        val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
        val_ds = val_ds.batch(32).prefetch(tf.data.AUTOTUNE)

    3️⃣  COMPILE MODEL
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

    4️⃣  TRAIN MODEL
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=20,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint('best_model.h5',
                                                   save_best_only=True),
                tf.keras.callbacks.EarlyStopping(patience=5),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
            ]
        )

    5️⃣  EVALUATE MODEL
        test_loss, test_acc = model.evaluate(test_ds)
        print(f"Test accuracy: {test_acc:.2%}")

    6️⃣  MAKE PREDICTIONS
        predictions = model.predict(new_images)
        predicted_classes = tf.argmax(predictions, axis=1)

    7️⃣  SAVE MODEL
        model.save_weights('final_model.h5')
        # or
        tf.saved_model.save(model, 'saved_model/')

    8️⃣  DEPLOY (Optional)
        # Convert to TFLite for mobile
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        # Or serve with TensorFlow Serving
        tf.saved_model.save(model, 'serving_model/1/')

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """

    print(workflow)

    # Demonstrate with actual code (simplified)
    print("\n[Executable Example]")
    print("Creating a complete minimal workflow...\n")

    # 1. Choose model
    print("1. Creating model...")
    model = create_vit_s16(image_size=224, include_top=True,
                           num_classes=5, dropout=0.1)
    print("   ✓ ViT-S/16 created")

    # 2. Prepare dummy data
    print("\n2. Preparing data...")
    train_images = tf.random.normal((100, 224, 224, 3))
    train_labels = tf.random.uniform((100,), maxval=5, dtype=tf.int32)

    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_ds = train_ds.batch(16).prefetch(tf.data.AUTOTUNE)
    print("   ✓ Training dataset prepared (100 samples)")

    # 3. Compile
    print("\n3. Compiling model...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    print("   ✓ Model compiled")

    # 4. Train (1 epoch for demo)
    print("\n4. Training model (1 epoch for demo)...")
    history = model.fit(train_ds, epochs=1, verbose=0)
    print(f"   ✓ Training complete - Loss: {history.history['loss'][0]:.4f}, "
          f"Accuracy: {history.history['accuracy'][0]:.2%}")

    # 5. Predict
    print("\n5. Making predictions...")
    test_image = tf.random.normal((1, 224, 224, 3))
    prediction = model(test_image, training=False)
    predicted_class = tf.argmax(tf.nn.softmax(prediction), axis=1)[0]
    print(f"   ✓ Predicted class: {predicted_class}")

    # 6. Save
    print("\n6. Saving model...")
    model.save_weights('demo_model_weights.h5')
    print("   ✓ Model saved to: demo_model_weights.h5")

    print("\n✅ Complete workflow demonstrated!")


# =============================================================================
# MAIN: Run All Examples
# =============================================================================

def main():
    """Run all examples."""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + " "*20 + "VISION TRANSFORMER - USAGE EXAMPLES" + " "*24 + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")

    print("\nThis script demonstrates all features of the ViT implementation.")
    print("It includes 10 comprehensive examples covering different use cases.\n")

    try:
        # Run all examples
        example_1_create_models()
        example_2_feature_extraction()
        example_3_classification()
        example_4_transfer_learning()
        example_5_different_image_sizes()
        example_6_compare_variants()
        example_7_save_load_models()
        example_8_batch_processing()
        example_9_model_selection_guide()
        example_10_complete_workflow()

        # Summary
        print("\n" + "="*80)
        print("🎉 ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nNext steps:")
        print("  1. Choose the right model variant for your task (see Example 9)")
        print("  2. Prepare your dataset")
        print("  3. Follow the complete workflow (see Example 10)")
        print("  4. Refer to MODEL_INSTANCES.md for advanced use cases")
        print("\nHappy coding with Vision Transformers! 🚀\n")

    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    # Run all examples
    main()
