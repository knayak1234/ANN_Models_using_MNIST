#!/bin/bash

# Enhanced installation script for MNIST Neural Network Project
# Compatible with Python 3.12 and modern systems

echo "🚀 Setting up MNIST Neural Network Project..."
echo "=============================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "📍 Detected Python version: $python_version"

# Upgrade pip first to avoid compatibility issues
echo "📦 Upgrading pip..."
python3 -m pip install --upgrade pip setuptools wheel

# Install dependencies with fallback options
echo "📦 Installing core dependencies..."

# Try installing with compatible versions
if python3 -m pip install "numpy>=1.26.0" "pandas>=2.0.0" "matplotlib>=3.7.0"; then
    echo "✅ Core dependencies installed successfully"
else
    echo "⚠️  Trying alternative core dependencies..."
    python3 -m pip install "numpy>=1.24.0,<2.0.0" "pandas>=1.5.0" "matplotlib>=3.5.0"
fi

# Install PyTorch (CPU version for broader compatibility)
echo "🔥 Installing PyTorch..."
if python3 -m pip install "torch>=2.1.0" "torchvision>=0.16.0" --index-url https://download.pytorch.org/whl/cpu; then
    echo "✅ PyTorch installed successfully"
else
    echo "⚠️  Trying alternative PyTorch installation..."
    python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# Install TensorFlow
echo "🧠 Installing TensorFlow..."
if python3 -m pip install "tensorflow>=2.15.0"; then
    echo "✅ TensorFlow installed successfully"
else
    echo "⚠️  Trying alternative TensorFlow installation..."
    python3 -m pip install "tensorflow>=2.13.0"
fi

# Install additional utilities
echo "🛠️  Installing additional utilities..."
python3 -m pip install "scikit-learn>=1.3.0" "seaborn>=0.12.0"

echo ""
echo "🎉 Installation completed!"
echo "=========================================="
echo ""
echo "🔍 Verifying installation..."
python3 -c "
import sys
print(f'Python version: {sys.version}')
try:
    import numpy as np
    print(f'✅ NumPy: {np.__version__}')
except ImportError as e:
    print(f'❌ NumPy: {e}')

try:
    import pandas as pd
    print(f'✅ Pandas: {pd.__version__}')
except ImportError as e:
    print(f'❌ Pandas: {e}')

try:
    import matplotlib
    print(f'✅ Matplotlib: {matplotlib.__version__}')
except ImportError as e:
    print(f'❌ Matplotlib: {e}')

try:
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
    print(f'   Device: {\"CUDA\" if torch.cuda.is_available() else \"CPU\"}')
except ImportError as e:
    print(f'❌ PyTorch: {e}')

try:
    import tensorflow as tf
    print(f'✅ TensorFlow: {tf.__version__}')
    print(f'   GPU Available: {len(tf.config.list_physical_devices(\"GPU\")) > 0}')
except ImportError as e:
    print(f'❌ TensorFlow: {e}')
"

echo ""
echo "🎯 Ready to run!"
echo "Try these commands:"
echo "  python3 artificial_neural_network_MNIST.py              # Original NumPy version"
echo "  python3 artificial_neural_network_MNIST_pytorch.py      # PyTorch version"
echo "  python3 artificial_neural_network_MNIST_tensorflow.py   # TensorFlow version"
echo "  python3 compare_all_implementations.py                  # Compare all three"
echo ""
echo "📁 Don't forget to download mnist_test.csv from Kaggle!"
