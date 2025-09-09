# 🚀 Setup Guide for MNIST Neural Network

This guide provides detailed step-by-step instructions to set up and run the artificial neural network project for MNIST digit classification.

## 📋 Table of Contents

1. [System Requirements](#system-requirements)
2. [Environment Setup](#environment-setup)
3. [Dataset Preparation](#dataset-preparation)
4. [Installation Steps](#installation-steps)
5. [Running the Project](#running-the-project)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Configuration](#advanced-configuration)

## 💻 System Requirements

### Minimum Requirements
- **Operating System**: Windows 10, macOS 10.12+, or Linux (Ubuntu 18.04+)
- **Python**: Version 3.7 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 500MB free space
- **CPU**: Any modern processor (multi-core recommended for faster training)

### Recommended Requirements
- **Python**: Version 3.8-3.11
- **RAM**: 8GB or more
- **Storage**: 1GB free space
- **CPU**: Multi-core processor for better performance

## 🔧 Environment Setup

### Option 1: Using System Python (Recommended for Beginners)

1. **Check Python Installation**
   ```bash
   python --version
   # or
   python3 --version
   ```
   
   If Python is not installed, download from [python.org](https://www.python.org/downloads/)

2. **Verify pip Installation**
   ```bash
   pip --version
   # or
   pip3 --version
   ```

### Option 2: Using Virtual Environment (Recommended for Development)

1. **Create Virtual Environment**
   ```bash
   # Navigate to project directory
   cd /path/to/first_ANN_model
   
   # Create virtual environment
   python -m venv venv
   
   # Alternative for Python 3
   python3 -m venv venv
   ```

2. **Activate Virtual Environment**
   
   **On Windows:**
   ```bash
   venv\Scripts\activate
   ```
   
   **On macOS/Linux:**
   ```bash
   source venv/bin/activate
   ```

3. **Verify Activation**
   ```bash
   which python
   # Should show path to venv/bin/python
   ```

### Option 3: Using Conda (For Data Science Workflows)

1. **Install Miniconda/Anaconda**
   - Download from [conda.io](https://docs.conda.io/en/latest/miniconda.html)

2. **Create Conda Environment**
   ```bash
   conda create -n mnist_nn python=3.9
   conda activate mnist_nn
   ```

## 📊 Dataset Preparation

### Download MNIST Dataset

1. **Go to Kaggle MNIST Dataset**
   - Visit: [https://www.kaggle.com/datasets/oddrationale/mnist-in-csv](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)

2. **Download Required Files**
   - `mnist_train.csv` (~109 MB)
   - `mnist_test.csv` (~18 MB)

3. **Alternative Download Methods**
   
   **Using Kaggle API (Advanced):**
   ```bash
   # Install Kaggle API
   pip install kaggle
   
   # Configure API credentials (requires Kaggle account)
   # Download dataset
   kaggle datasets download -d oddrationale/mnist-in-csv
   unzip mnist-in-csv.zip
   ```

4. **File Placement**
   ```
   first_ANN_model/
   ├── artificial_neural_network_MNIST.py
   ├── mnist_train.csv          # ← Place here
   ├── mnist_test.csv           # ← Place here
   ├── requirements.txt
   └── README.md
   ```

### Verify Dataset Files

Check file sizes to ensure complete downloads:
- `mnist_train.csv`: ~109 MB (60,000 samples)
- `mnist_test.csv`: ~18 MB (10,000 samples)

## 🔨 Installation Steps

### Step 1: Clone/Download Project

```bash
# If using Git
git clone <repository-url>
cd first_ANN_model

# Or download and extract ZIP file
# Navigate to extracted folder
```

### Step 2: Install Dependencies

```bash
# Make sure you're in the project directory
cd first_ANN_model

# Install required packages
pip install -r requirements.txt

# Verify installation
pip list
```

**Expected packages:**
- `numpy==1.24.3`
- `pandas==2.0.3`
- `matplotlib==3.7.2`

### Step 3: Verify Installation

Create a test script to verify everything works:

```python
# test_installation.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("✅ NumPy version:", np.__version__)
print("✅ Pandas version:", pd.__version__)
print("✅ Matplotlib version:", plt.matplotlib.__version__)
print("🎉 All dependencies installed successfully!")
```

Run the test:
```bash
python test_installation.py
```

## ▶️ Running the Project

### Basic Execution

1. **Navigate to Project Directory**
   ```bash
   cd /path/to/first_ANN_model
   ```

2. **Run the Neural Network**
   ```bash
   python artificial_neural_network_MNIST.py
   ```

### Expected Execution Flow

1. **Data Loading** (10-30 seconds)
   ```
   Original data shape: (10000, 785)
   Data dimensions: 10000 samples, 785 features
   ```

2. **Training Progress** (2-5 minutes)
   ```
   Iteration: 0
   Accuracy: 0.1123
   Iteration: 10
   Accuracy: 0.3456
   ...
   ```

3. **Results Display**
   ```
   Development set accuracy: 0.8523
   Sample predictions:
   Prediction:  [7]
   Label:  7
   ```

4. **Image Visualization**
   - matplotlib windows will open showing digit images
   - Close each window to see the next prediction

## 🔍 Troubleshooting

### Common Issues and Solutions

#### 1. **ModuleNotFoundError**
```
ModuleNotFoundError: No module named 'numpy'
```

**Solution:**
```bash
pip install numpy pandas matplotlib
# or
pip install -r requirements.txt
```

#### 2. **FileNotFoundError**
```
FileNotFoundError: [Errno 2] No such file or directory: 'mnist_test.csv'
```

**Solutions:**
- Ensure MNIST CSV files are in the project directory
- Check file names are exactly `mnist_train.csv` and `mnist_test.csv`
- Verify files downloaded completely

#### 3. **Memory Error**
```
MemoryError: Unable to allocate array
```

**Solutions:**
- Close other applications to free RAM
- Use a machine with more memory
- Reduce dataset size by modifying the code

#### 4. **Permission Denied**
```
PermissionError: [Errno 13] Permission denied
```

**Solutions:**
```bash
# On macOS/Linux
sudo python artificial_neural_network_MNIST.py

# Or fix file permissions
chmod +x artificial_neural_network_MNIST.py
```

#### 5. **Matplotlib Display Issues**

**On Linux (headless servers):**
```python
# Add this before importing matplotlib
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
```

**On macOS with virtual environments:**
```bash
# Install with specific backend
pip install matplotlib --force-reinstall --no-deps
```

#### 6. **Slow Training**

**Solutions:**
- Reduce number of iterations (line 111)
- Use fewer training samples
- Run on a faster machine

### Debugging Tips

1. **Check Data Loading**
   ```python
   # Add after line 12
   print("First few samples:", data[:5, :5])
   ```

2. **Monitor Memory Usage**
   ```python
   import psutil
   print(f"Memory usage: {psutil.virtual_memory().percent}%")
   ```

3. **Verify Data Shapes**
   ```python
   # Add debugging prints
   print(f"X_train shape: {X_train.shape}")
   print(f"Y_train shape: {Y_train.shape}")
   ```

## ⚙️ Advanced Configuration

### Modifying Training Parameters

```python
# In artificial_neural_network_MNIST.py

# Line 111: Adjust learning rate and iterations
W1, b1, W2, b2 = gradient_descent(X_train_norm, Y_train_one_hot, 0.1, 500)
#                                                                    ^^^  ^^^
#                                                             learning_rate iterations

# Line 18-19: Adjust data split
data_dev = data[0:1000].T    # Development samples
data_train = data[1000:m].T  # Training samples

# Lines 34-37: Modify network architecture
W1 = np.random.rand(10, 784) - 0.5  # Hidden layer size
```

### Performance Optimization

1. **Use Different Random Seed**
   ```python
   # Add at the beginning
   np.random.seed(42)
   ```

2. **Adjust Batch Processing**
   ```python
   # Process in smaller batches for memory efficiency
   batch_size = 1000
   ```

3. **Early Stopping**
   ```python
   # Stop when accuracy doesn't improve
   if accuracy > 0.95:
       break
   ```

### Custom Modifications

1. **Add More Hidden Layers**
   ```python
   # Modify init_params() and forward_prop()
   W1 = np.random.rand(128, 784) - 0.5  # First hidden layer
   W2 = np.random.rand(64, 128) - 0.5   # Second hidden layer
   W3 = np.random.rand(10, 64) - 0.5    # Output layer
   ```

2. **Different Activation Functions**
   ```python
   def sigmoid(z):
       return 1 / (1 + np.exp(-z))
   
   def tanh(z):
       return np.tanh(z)
   ```

3. **Learning Rate Decay**
   ```python
   # Reduce learning rate over time
   alpha = 0.1 * (0.95 ** (i // 100))
   ```

## 📈 Performance Benchmarks

### Expected Results
- **Training Time**: 2-5 minutes (500 iterations)
- **Memory Usage**: 2-4 GB RAM
- **Accuracy**: 85-92% on development set
- **File Sizes**: 
  - Training data: ~109 MB
  - Test data: ~18 MB

### Hardware Performance
- **Intel i5 (4 cores)**: ~3 minutes
- **Intel i7 (8 cores)**: ~2 minutes
- **Apple M1**: ~1.5 minutes
- **Google Colab (free)**: ~2 minutes

## 🔗 Additional Resources

### Learning Materials
- [3Blue1Brown Neural Networks](https://www.3blue1brown.com/topics/neural-networks)
- [Andrew Ng's Machine Learning Course](https://www.coursera.org/learn/machine-learning)
- [Fast.ai Practical Deep Learning](https://course.fast.ai/)

### Development Tools
- **Jupyter Notebook**: Interactive development
- **Google Colab**: Free GPU/TPU access
- **PyCharm**: Professional Python IDE
- **VS Code**: Lightweight editor with Python support

### Next Steps
1. Try different network architectures
2. Implement other optimization algorithms (Adam, RMSprop)
3. Add regularization techniques
4. Explore convolutional neural networks
5. Work with other datasets (CIFAR-10, Fashion-MNIST)

---

**Need help?** Check the main README.md or create an issue in the repository!

**Happy coding! 🎉**
