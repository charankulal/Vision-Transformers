#!/bin/bash
# Helper script for Linux/Mac users to upload to Kaggle

echo "==============================================================="
echo "Vision Transformer - Kaggle Upload (Linux/Mac)"
echo "==============================================================="
echo ""

# Check if username is provided
if [ -z "$1" ]; then
    echo "Usage: ./upload_to_kaggle.sh YOUR_KAGGLE_USERNAME [--update]"
    echo ""
    echo "Examples:"
    echo "  ./upload_to_kaggle.sh john_doe"
    echo "  ./upload_to_kaggle.sh john_doe --update"
    echo ""
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    echo "Please install Python from https://www.python.org/"
    exit 1
fi

# Run the Python script
if [ "$2" == "--update" ]; then
    python3 upload_to_kaggle.py --username "$1" --update
else
    python3 upload_to_kaggle.py --username "$1"
fi
