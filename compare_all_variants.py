"""
Compare and benchmark all Vision Transformer variants
"""

import tensorflow as tf
import numpy as np
import time
from vit_all_variants import create_vit, VIT_CONFIGS


def benchmark_inference_speed(model, image_size, num_iterations=10, warmup=2):
    """
    Benchmark inference speed of a model.

    Args:
        model: The model to benchmark
        image_size: Input image size
        num_iterations: Number of iterations for timing
        warmup: Number of warmup iterations

    Returns:
        Average inference time in milliseconds
    """
    dummy_input = tf.random.normal((1, image_size, image_size, 3))

    # Warmup
    for _ in range(warmup):
        _ = model(dummy_input, training=False)

    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.time()
        _ = model(dummy_input, training=False)
        end = time.time()
        times.append((end - start) * 1000)  # Convert to ms

    return np.mean(times), np.std(times)


def detailed_comparison():
    """Provide detailed comparison of all variants."""
    print("\n" + "="*120)
    print("Detailed Vision Transformer Variant Comparison")
    print("="*120 + "\n")

    results = []

    for variant_name, config in VIT_CONFIGS.items():
        print(f"Analyzing {variant_name.upper()}...")

        # Create model
        model = create_vit(variant_name, image_size=224, include_top=False)

        # Calculate statistics
        params = model.count_params()
        num_patches = (224 // config['patch_size']) ** 2
        num_tokens = num_patches + 1

        # Estimate memory (rough approximation)
        memory_mb = (params * 4) / (1024 * 1024)  # 4 bytes per parameter (float32)

        results.append({
            'variant': variant_name.upper(),
            'patch_size': config['patch_size'],
            'layers': config['num_layers'],
            'embed_dim': config['embed_dim'],
            'heads': config['num_heads'],
            'mlp_dim': config['mlp_dim'],
            'params': params,
            'num_patches': num_patches,
            'num_tokens': num_tokens,
            'memory_mb': memory_mb
        })

    # Print table
    print("\n" + "="*120)
    header = f"{'Variant':<10} {'Patch':<7} {'Layers':<7} {'Embed':<7} {'Heads':<7} {'MLP':<7} {'Params':<15} {'Patches':<8} {'Tokens':<8} {'Memory':<10}"
    print(header)
    print("="*120)

    for r in results:
        print(f"{r['variant']:<10} "
              f"{r['patch_size']:<7} "
              f"{r['layers']:<7} "
              f"{r['embed_dim']:<7} "
              f"{r['heads']:<7} "
              f"{r['mlp_dim']:<7} "
              f"{r['params']:>12,}   "
              f"{r['num_patches']:<8} "
              f"{r['num_tokens']:<8} "
              f"{r['memory_mb']:>6.1f} MB")

    print("="*120 + "\n")


def speed_benchmark(image_size=224):
    """Benchmark inference speed for all variants."""
    print("\n" + "="*100)
    print(f"Inference Speed Benchmark (Image Size: {image_size}x{image_size})")
    print("="*100 + "\n")

    print("Warming up and benchmarking models...\n")

    results = []

    for variant_name in VIT_CONFIGS.keys():
        print(f"Benchmarking {variant_name.upper()}...", end=" ")

        model = create_vit(variant_name, image_size=image_size, include_top=False)
        avg_time, std_time = benchmark_inference_speed(model, image_size)

        results.append({
            'variant': variant_name.upper(),
            'avg_time': avg_time,
            'std_time': std_time,
            'throughput': 1000 / avg_time  # images per second
        })

        print(f"{avg_time:.2f}ms")

    # Sort by speed (fastest first)
    results.sort(key=lambda x: x['avg_time'])

    # Print table
    print("\n" + "="*100)
    print(f"{'Variant':<12} {'Avg Time (ms)':<18} {'Std Dev (ms)':<18} {'Throughput (img/s)':<20} {'Speed':<10}")
    print("="*100)

    baseline_time = results[0]['avg_time']

    for r in results:
        speedup = baseline_time / r['avg_time']
        speed_indicator = "⚡ Fastest" if speedup >= 0.95 else f"{speedup:.2f}x slower"

        print(f"{r['variant']:<12} "
              f"{r['avg_time']:>10.2f}         "
              f"{r['std_time']:>10.2f}         "
              f"{r['throughput']:>10.2f}           "
              f"{speed_indicator}")

    print("="*100 + "\n")


def memory_analysis():
    """Analyze memory requirements for all variants."""
    print("\n" + "="*80)
    print("Memory Requirements Analysis")
    print("="*80 + "\n")

    results = []

    for variant_name, config in VIT_CONFIGS.items():
        model = create_vit(variant_name, image_size=224, include_top=False)

        params = model.count_params()
        memory_params = (params * 4) / (1024 * 1024)  # Parameters (float32)
        memory_activations = (224 * 224 * 3 * 4) / (1024 * 1024)  # Input
        num_tokens = ((224 // config['patch_size']) ** 2) + 1
        memory_features = (num_tokens * config['embed_dim'] * 4) / (1024 * 1024)  # Output

        total_memory = memory_params + memory_activations + memory_features

        results.append({
            'variant': variant_name.upper(),
            'params': params,
            'memory_params': memory_params,
            'memory_activations': memory_activations,
            'memory_features': memory_features,
            'total_memory': total_memory
        })

    # Print table
    print(f"{'Variant':<12} {'Parameters':<15} {'Param Mem':<12} {'Input Mem':<12} {'Output Mem':<12} {'Total Mem':<12}")
    print("="*80)

    for r in results:
        print(f"{r['variant']:<12} "
              f"{r['params']:>12,}   "
              f"{r['memory_params']:>8.1f} MB   "
              f"{r['memory_activations']:>8.1f} MB   "
              f"{r['memory_features']:>8.1f} MB   "
              f"{r['total_memory']:>8.1f} MB")

    print("="*80 + "\n")


def use_case_recommendations():
    """Provide recommendations for different use cases."""
    print("\n" + "="*80)
    print("Use Case Recommendations")
    print("="*80 + "\n")

    recommendations = {
        "Real-time Applications (Mobile/Edge)": {
            "models": ["ViT-S/32"],
            "reason": "Smallest model, fastest inference, suitable for resource-constrained devices"
        },
        "Balanced Performance": {
            "models": ["ViT-S/16", "ViT-B/32"],
            "reason": "Good trade-off between speed and accuracy"
        },
        "High Accuracy (Cloud/Server)": {
            "models": ["ViT-B/16", "ViT-L/16"],
            "reason": "Better feature extraction with reasonable compute requirements"
        },
        "State-of-the-art Performance": {
            "models": ["ViT-L/16", "ViT-H/14"],
            "reason": "Highest accuracy, suitable for research and high-end applications"
        },
        "Feature Extraction": {
            "models": ["ViT-B/16", "ViT-B/32"],
            "reason": "Standard choice for transfer learning and embeddings"
        },
        "Fine-tuning on Small Datasets": {
            "models": ["ViT-S/16", "ViT-S/32"],
            "reason": "Less prone to overfitting, faster training"
        },
        "Large-scale Training": {
            "models": ["ViT-L/16", "ViT-H/14"],
            "reason": "Better capacity for learning from massive datasets"
        }
    }

    for use_case, info in recommendations.items():
        print(f"📌 {use_case}")
        print(f"   Recommended: {', '.join(info['models'])}")
        print(f"   Reason: {info['reason']}")
        print()


def accuracy_vs_speed_analysis():
    """Analyze the accuracy vs speed trade-offs."""
    print("\n" + "="*80)
    print("Accuracy vs Speed Trade-offs")
    print("="*80 + "\n")

    print("General Patterns:\n")

    print("1. Model Size Impact:")
    print("   Small (S) → Base (B) → Large (L) → Huge (H)")
    print("   - Accuracy: ↑↑↑ (increases significantly)")
    print("   - Speed: ↓↓↓ (decreases significantly)")
    print("   - Memory: ↑↑↑ (increases significantly)")
    print()

    print("2. Patch Size Impact (within same model size):")
    print("   /32 → /16 → /14")
    print("   - Accuracy: ↑ (increases moderately)")
    print("   - Speed: ↓↓ (decreases significantly)")
    print("   - Tokens: ↑↑↑ (increases quadratically)")
    print()

    print("3. Speed Ranking (Fastest → Slowest):")
    print("   ViT-S/32 > ViT-S/16 > ViT-B/32 > ViT-B/16 > ViT-L/32 > ViT-L/16 > ViT-H/16 > ViT-H/14")
    print()

    print("4. Typical ImageNet-1K Accuracy (Top-1 %):")
    print("   ViT-S/32: ~75%")
    print("   ViT-S/16: ~78%")
    print("   ViT-B/32: ~75-77%")
    print("   ViT-B/16: ~80-82%")
    print("   ViT-L/16: ~83-85%")
    print("   ViT-H/14: ~86-88%")
    print()


def output_shape_examples():
    """Show output shapes for different configurations."""
    print("\n" + "="*100)
    print("Output Shape Examples (Batch Size = 4)")
    print("="*100 + "\n")

    test_input = tf.random.normal((4, 224, 224, 3))

    print(f"{'Variant':<12} {'Image Size':<12} {'Patch Grid':<12} {'Num Tokens':<12} {'Output Shape':<30}")
    print("="*100)

    for variant_name, config in VIT_CONFIGS.items():
        model = create_vit(variant_name, image_size=224, include_top=False)
        output = model(test_input, training=False)

        grid_size = 224 // config['patch_size']
        num_patches = grid_size * grid_size

        print(f"{variant_name.upper():<12} "
              f"224x224      "
              f"{grid_size}x{grid_size}        "
              f"{num_patches + 1:<12} "
              f"{str(tuple(output.shape)):<30}")

    print("="*100 + "\n")
    print("Note: Output shape is (batch_size, num_tokens, embed_dim)")
    print("      num_tokens = num_patches + 1 (class token)")
    print()


def main():
    """Run all comparisons and analyses."""
    print("\n" + "🔍 "*40)
    print("VISION TRANSFORMER VARIANTS - COMPREHENSIVE ANALYSIS")
    print("🔍 "*40)

    # Detailed comparison
    detailed_comparison()

    # Speed benchmark
    speed_benchmark(image_size=224)

    # Memory analysis
    memory_analysis()

    # Output shapes
    output_shape_examples()

    # Accuracy vs speed
    accuracy_vs_speed_analysis()

    # Recommendations
    use_case_recommendations()

    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
