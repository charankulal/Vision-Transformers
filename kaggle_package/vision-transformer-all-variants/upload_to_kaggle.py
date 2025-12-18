#!/usr/bin/env python3
"""
Helper script to upload Vision Transformer dataset to Kaggle

Usage:
    python upload_to_kaggle.py --username your_kaggle_username
    python upload_to_kaggle.py --username your_kaggle_username --update
    python upload_to_kaggle.py --username your_kaggle_username --update --message "Version 2.0"
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def check_kaggle_api():
    """Check if Kaggle API is installed."""
    try:
        import kaggle
        return True
    except ImportError:
        return False


def check_kaggle_credentials():
    """Check if Kaggle API credentials are configured."""
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'

    if not kaggle_json.exists():
        # Try Windows location
        kaggle_dir = Path(os.environ.get('USERPROFILE', '')) / '.kaggle'
        kaggle_json = kaggle_dir / 'kaggle.json'

    return kaggle_json.exists()


def update_metadata(username):
    """Update dataset-metadata.json with the correct username."""
    metadata_file = Path('dataset-metadata.json')

    if not metadata_file.exists():
        print("❌ Error: dataset-metadata.json not found!")
        print("   Make sure you're running this script from the dataset directory.")
        return False

    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Update the ID with the correct username
        old_id = metadata.get('id', 'yourusername/vision-transformer-all-variants')
        new_id = f"{username}/vision-transformer-all-variants"
        metadata['id'] = new_id

        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"✓ Updated dataset ID: {old_id} → {new_id}")
        return True

    except Exception as e:
        print(f"❌ Error updating metadata: {e}")
        return False


def verify_files():
    """Verify all required files are present."""
    required_files = [
        'dataset-metadata.json',
        'README.md',
        'requirements.txt',
        'src/__init__.py',
        'src/vit_all_variants.py',
    ]

    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False

    print("✓ All required files present")
    return True


def create_dataset():
    """Create a new dataset on Kaggle."""
    print("\n📦 Creating new dataset on Kaggle...")
    try:
        result = subprocess.run(
            ['kaggle', 'datasets', 'create', '-p', '.'],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        print("\n✅ Dataset created successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error creating dataset:")
        print(e.stderr)
        return False
    except FileNotFoundError:
        print("❌ Kaggle CLI not found. Is it installed?")
        print("   Install with: pip install kaggle")
        return False


def update_dataset(message):
    """Update an existing dataset on Kaggle."""
    print(f"\n🔄 Updating dataset on Kaggle...")
    if message:
        print(f"   Version message: {message}")

    try:
        cmd = ['kaggle', 'datasets', 'version', '-p', '.']
        if message:
            cmd.extend(['-m', message])
        else:
            cmd.extend(['-m', 'Updated dataset'])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        print("\n✅ Dataset updated successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error updating dataset:")
        print(e.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Upload Vision Transformer dataset to Kaggle',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # First time upload
  python upload_to_kaggle.py --username john_doe

  # Update existing dataset
  python upload_to_kaggle.py --username john_doe --update

  # Update with custom message
  python upload_to_kaggle.py --username john_doe --update --message "Added new features"
        """
    )

    parser.add_argument(
        '--username',
        required=True,
        help='Your Kaggle username'
    )

    parser.add_argument(
        '--update',
        action='store_true',
        help='Update existing dataset (default: create new)'
    )

    parser.add_argument(
        '--message',
        help='Version message for update (optional)'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Vision Transformer - Kaggle Dataset Upload Tool")
    print("=" * 70)

    # Check Kaggle API
    print("\n1️⃣ Checking Kaggle API...")
    if not check_kaggle_api():
        print("❌ Kaggle API not installed!")
        print("   Install with: pip install kaggle")
        sys.exit(1)
    print("✓ Kaggle API installed")

    # Check credentials
    print("\n2️⃣ Checking Kaggle credentials...")
    if not check_kaggle_credentials():
        print("❌ Kaggle API credentials not found!")
        print("   Download kaggle.json from: https://www.kaggle.com/settings")
        print("   Place it in: ~/.kaggle/kaggle.json (Linux/Mac)")
        print("            or: %USERPROFILE%\\.kaggle\\kaggle.json (Windows)")
        sys.exit(1)
    print("✓ Kaggle credentials found")

    # Verify files
    print("\n3️⃣ Verifying files...")
    if not verify_files():
        sys.exit(1)

    # Update metadata
    print("\n4️⃣ Updating metadata...")
    if not update_metadata(args.username):
        sys.exit(1)

    # Upload or update
    if args.update:
        print("\n5️⃣ Updating dataset...")
        success = update_dataset(args.message)
    else:
        print("\n5️⃣ Creating new dataset...")
        success = create_dataset()

    if success:
        print("\n" + "=" * 70)
        print("🎉 Success!")
        print("=" * 70)
        print(f"\nYour dataset is available at:")
        print(f"https://www.kaggle.com/datasets/{args.username}/vision-transformer-all-variants")
        print("\nNext steps:")
        print("1. Visit the dataset URL above")
        print("2. Verify all files are present")
        print("3. Create a notebook using your dataset")
        print("4. Share with the community!")
    else:
        print("\n" + "=" * 70)
        print("❌ Upload failed")
        print("=" * 70)
        print("\nPlease check the error messages above and try again.")
        print("For help, see: KAGGLE_UPLOAD_GUIDE.md")
        sys.exit(1)


if __name__ == '__main__':
    main()
