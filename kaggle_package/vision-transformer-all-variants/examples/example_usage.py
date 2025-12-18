"""
Example usage of Vision Transformer B32 Feature Extractor
"""

import tensorflow as tf
import numpy as np
from vit_all_variants import create_vit_b32


def example_feature_extraction():
    """
    Example 1: Use ViT-B32 as a feature extractor
    """
    print("=" * 80)
    print("Example 1: Feature Extraction")
    print("=" * 80)

    # Create feature extractor
    model = create_vit_b32(
        image_size=224,
        include_top=False  # No classification head
    )

    # Simulate batch of images
    images = tf.random.normal((4, 224, 224, 3))

    # Extract features (all tokens)
    all_features = model(images, training=False)
    print(f"All tokens shape: {all_features.shape}")
    print(f"-> (batch_size=4, num_tokens=50, embed_dim=768)")

    # Extract class token features only
    cls_features = model.extract_features(images, training=False)
    print(f"\nClass token features shape: {cls_features.shape}")
    print(f"-> (batch_size=4, embed_dim=768)")
    print("\nThese 768-dimensional vectors can be used for:")
    print("  - Image similarity search")
    print("  - Transfer learning")
    print("  - Image clustering")
    print("  - Fine-tuning on downstream tasks")


def example_classification():
    """
    Example 2: Use ViT-B32 with classification head
    """
    print("\n" + "=" * 80)
    print("Example 2: Image Classification")
    print("=" * 80)

    num_classes = 10  # e.g., CIFAR-10

    # Create model with classification head
    model = create_vit_b32(
        image_size=224,
        include_top=True,
        num_classes=num_classes
    )

    # Simulate batch of images
    images = tf.random.normal((4, 224, 224, 3))

    # Get predictions
    logits = model(images, training=False)
    probabilities = tf.nn.softmax(logits)

    print(f"Logits shape: {logits.shape}")
    print(f"-> (batch_size=4, num_classes={num_classes})")
    print(f"\nSample predictions for first image:")
    print(f"Probabilities: {probabilities[0].numpy()}")
    print(f"Predicted class: {tf.argmax(probabilities[0]).numpy()}")


def example_fine_tuning():
    """
    Example 3: Fine-tune ViT-B32 on custom dataset
    """
    print("\n" + "=" * 80)
    print("Example 3: Fine-tuning Setup")
    print("=" * 80)

    num_classes = 5  # Custom dataset classes

    # Create model
    model = create_vit_b32(
        image_size=224,
        include_top=True,
        num_classes=num_classes
    )

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    print("Model compiled and ready for training!")
    print("\nTraining setup:")
    print("  - Optimizer: Adam (lr=1e-4)")
    print("  - Loss: SparseCategoricalCrossentropy")
    print("  - Metrics: Accuracy")

    # Simulate training data
    print("\nCreating dummy dataset for demonstration...")
    dummy_images = tf.random.normal((32, 224, 224, 3))
    dummy_labels = tf.random.uniform((32,), minval=0, maxval=num_classes, dtype=tf.int32)

    # Single training step (for demonstration)
    print("\nRunning single training step...")
    history = model.fit(
        dummy_images,
        dummy_labels,
        batch_size=8,
        epochs=1,
        verbose=1
    )

    print(f"\nTraining loss: {history.history['loss'][0]:.4f}")
    print(f"Training accuracy: {history.history['accuracy'][0]:.4f}")


def example_custom_image_size():
    """
    Example 4: Use different input sizes
    """
    print("\n" + "=" * 80)
    print("Example 4: Custom Image Sizes")
    print("=" * 80)

    # ViT-B32 with patch_size=32 works with any image size divisible by 32
    image_sizes = [224, 256, 384]

    for img_size in image_sizes:
        model = create_vit_b32(image_size=img_size, include_top=False)

        # Test with corresponding image size
        test_image = tf.random.normal((1, img_size, img_size, 3))
        features = model(test_image, training=False)

        num_patches = (img_size // 32) ** 2
        print(f"\nImage size: {img_size}x{img_size}")
        print(f"  Number of patches: {num_patches}")
        print(f"  Output shape: {features.shape}")
        print(f"  Expected: (1, {num_patches + 1}, 768)")


def example_save_load_model():
    """
    Example 5: Save and load model
    """
    print("\n" + "=" * 80)
    print("Example 5: Save and Load Model")
    print("=" * 80)

    # Create and save model
    print("Creating model...")
    model = create_vit_b32(image_size=224, include_top=False)

    save_path = "vit_b32_saved_model"
    print(f"\nSaving model to: {save_path}")
    model.save_weights(save_path)
    print("Model saved successfully!")

    # Load model
    print("\nLoading model...")
    new_model = create_vit_b32(
        image_size=224,
        include_top=False,
        weights=save_path
    )
    print("Model loaded successfully!")

    # Verify they produce same outputs
    test_image = tf.random.normal((1, 224, 224, 3))
    original_output = model(test_image, training=False)
    loaded_output = new_model(test_image, training=False)

    print(f"\nOutput difference (should be ~0): {tf.reduce_max(tf.abs(original_output - loaded_output)).numpy():.10f}")


def example_transfer_learning():
    """
    Example 6: Transfer learning with frozen backbone
    """
    print("\n" + "=" * 80)
    print("Example 6: Transfer Learning with Frozen Backbone")
    print("=" * 80)

    # Create base model
    base_model = create_vit_b32(image_size=224, include_top=False)

    # Freeze the base model
    base_model.trainable = False

    # Create new model with custom head
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)

    # Take class token
    x = x[:, 0, :]

    # Add custom classification layers
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

    # Create final model
    transfer_model = tf.keras.Model(inputs=inputs, outputs=outputs)

    print("Transfer learning model created!")
    print(f"\nBase model trainable: {base_model.trainable}")
    print(f"Total parameters: {transfer_model.count_params():,}")
    print(f"Trainable parameters: {sum([tf.size(w).numpy() for w in transfer_model.trainable_weights]):,}")
    print(f"Non-trainable parameters: {sum([tf.size(w).numpy() for w in transfer_model.non_trainable_weights]):,}")

    transfer_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\nModel ready for training with frozen ViT backbone!")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Vision Transformer B32 - Example Usage")
    print("=" * 80 + "\n")

    # Run all examples
    example_feature_extraction()
    example_classification()
    example_fine_tuning()
    example_custom_image_size()
    example_save_load_model()
    example_transfer_learning()

    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80 + "\n")
