# Kaggle Package - Quick Start Guide

This guide will help you upload the Vision Transformer implementation to Kaggle in just a few steps.

## 🎯 Goal

Upload your Vision Transformer models as a Kaggle Dataset so others can use them in notebooks and competitions.

## ⚡ Quick Upload (3 Steps)

### Step 1: Install Kaggle API

```bash
pip install kaggle
```

### Step 2: Setup Credentials

1. Go to https://www.kaggle.com/settings
2. Scroll to "API" section
3. Click "Create New API Token"
4. Download `kaggle.json`
5. Place it in:
   - **Windows**: `C:\Users\YourName\.kaggle\kaggle.json`
   - **Linux/Mac**: `~/.kaggle/kaggle.json`

### Step 3: Upload to Kaggle

**Windows:**
```bash
cd kaggle_package\vision-transformer-all-variants
upload_to_kaggle.bat YOUR_KAGGLE_USERNAME
```

**Linux/Mac:**
```bash
cd kaggle_package/vision-transformer-all-variants
chmod +x upload_to_kaggle.sh
./upload_to_kaggle.sh YOUR_KAGGLE_USERNAME
```

**Or use Python directly:**
```bash
cd kaggle_package/vision-transformer-all-variants
python upload_to_kaggle.py --username YOUR_KAGGLE_USERNAME
```

That's it! Your dataset will be live at:
`https://www.kaggle.com/datasets/YOUR_KAGGLE_USERNAME/vision-transformer-all-variants`

## 🔄 Updating Your Dataset

After making changes, upload a new version:

**Windows:**
```bash
upload_to_kaggle.bat YOUR_KAGGLE_USERNAME --update
```

**Linux/Mac:**
```bash
./upload_to_kaggle.sh YOUR_KAGGLE_USERNAME --update
```

**Python:**
```bash
python upload_to_kaggle.py --username YOUR_KAGGLE_USERNAME --update --message "Added new features"
```

## 📦 What Gets Uploaded

```
vision-transformer-all-variants/
├── src/
│   ├── __init__.py               # Package init with exports
│   └── vit_all_variants.py       # Main implementation (17KB)
├── examples/
│   ├── example_usage.py          # Python usage examples
│   └── kaggle_quickstart.ipynb   # Kaggle notebook
├── README.md                     # Kaggle-optimized docs (16KB)
├── requirements.txt              # TensorFlow + NumPy
└── dataset-metadata.json         # Kaggle metadata
```

## ✅ Verify Upload

1. Visit your dataset URL
2. Check all files are present
3. Create a test notebook:

```python
import sys
sys.path.append('/kaggle/input/vision-transformer-all-variants/src')

from vit_all_variants import create_vit_b16
model = create_vit_b16(image_size=224, include_top=False)
print(f"✓ Model created! Parameters: {model.count_params():,}")
```

## 🎓 Using Your Dataset in Kaggle Notebooks

1. Create a new notebook on Kaggle
2. Click "Add Data" → Search for "vision-transformer-all-variants"
3. Add your dataset
4. Use the import code above

## 📖 Full Documentation

For detailed instructions, see:
- `vision-transformer-all-variants/KAGGLE_UPLOAD_GUIDE.md` - Complete guide
- `vision-transformer-all-variants/README.md` - Usage documentation
- `vision-transformer-all-variants/examples/` - Example code

## 🆘 Troubleshooting

**Problem**: "Kaggle API credentials not found"
**Solution**: Make sure `kaggle.json` is in `~/.kaggle/` directory with correct permissions

**Problem**: "Dataset already exists"
**Solution**: Use `--update` flag to create a new version

**Problem**: "Import error in Kaggle notebook"
**Solution**: Verify the path matches your dataset name:
```python
sys.path.append('/kaggle/input/vision-transformer-all-variants/src')
```

## 🎉 After Upload

1. Share your dataset link with the community
2. Create example notebooks showing use cases
3. Use in Kaggle competitions
4. Get feedback and iterate

## 📞 Need Help?

- Full guide: `vision-transformer-all-variants/KAGGLE_UPLOAD_GUIDE.md`
- Kaggle API docs: https://github.com/Kaggle/kaggle-api
- Kaggle forums: https://www.kaggle.com/discussion

---

**Ready to upload?** Run the upload script now! 🚀
