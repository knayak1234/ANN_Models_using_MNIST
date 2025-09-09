# 📊 MNIST Dataset Guide

This guide explains how to obtain and prepare the MNIST dataset for the neural network implementations.

## 🎯 About the MNIST Dataset

The **MNIST (Modified National Institute of Standards and Technology)** database is a large collection of handwritten digits commonly used for training and testing machine learning systems.

### 📈 Dataset Statistics
- **Training samples**: 60,000 images
- **Test samples**: 10,000 images  
- **Image dimensions**: 28×28 pixels (grayscale)
- **Classes**: 10 digits (0-9)
- **File format**: CSV (Comma Separated Values)
- **Total size**: ~127 MB

## 📥 How to Download

### Option 1: Kaggle (Recommended)

1. **Visit the Kaggle Dataset**:
   - Go to: [https://www.kaggle.com/datasets/oddrationale/mnist-in-csv](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)

2. **Download the Files**:
   - Click the "Download" button
   - Extract the ZIP file
   - You'll get two CSV files:
     - `mnist_train.csv` (~109 MB)
     - `mnist_test.csv` (~18 MB)

3. **Place in Project Directory**:
   ```bash
   # Move files to your project folder
   mv path/to/downloads/mnist_*.csv /path/to/your/project/
   ```

### Option 2: Kaggle API (Advanced)

```bash
# Install Kaggle API
pip install kaggle

# Configure API credentials (requires Kaggle account)
# 1. Go to Kaggle.com → Account → API → Create New API Token
# 2. Download kaggle.json and place in ~/.kaggle/

# Download dataset
kaggle datasets download -d oddrationale/mnist-in-csv

# Extract files
unzip mnist-in-csv.zip
```

### Option 3: Direct Links (Alternative)

If Kaggle is not accessible, you can find MNIST CSV versions on:
- [Official MNIST](http://yann.lecun.com/exdb/mnist/) (binary format, needs conversion)
- [Git LFS repositories](https://github.com/search?q=mnist+csv) (search for CSV versions)

## 📋 File Structure

After downloading, your project directory should look like:

```
your-project/
├── mnist_train.csv          # ← Training data (60,000 samples)
├── mnist_test.csv           # ← Test data (10,000 samples)
├── artificial_neural_network_MNIST.py
├── artificial_neural_network_MNIST_pytorch.py
├── artificial_neural_network_MNIST_tensorflow.py
└── other project files...
```

## 🔍 Understanding the CSV Format

### File Structure
Each CSV file contains:
- **First column**: Label (digit 0-9)
- **Remaining 784 columns**: Pixel values (0-255)

### Example Row
```csv
label,pixel0,pixel1,pixel2,...,pixel783
5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,...
```

### Data Interpretation
- **Label**: The actual digit (0, 1, 2, ..., 9)
- **Pixels**: 28×28 = 784 pixel values
- **Pixel values**: 0 (black) to 255 (white)
- **Image reconstruction**: Reshape 784 values to 28×28 matrix

## 🛠️ Data Preprocessing

### How the Code Handles Data

#### 1. **Loading** (all implementations)
```python
import pandas as pd
data = pd.read_csv("mnist_test.csv")
```

#### 2. **Splitting** (features vs labels)
```python
# NumPy version
Y = data_array[:, 0]      # First column (labels)
X = data_array[:, 1:]     # Remaining columns (pixels)

# PyTorch/TensorFlow versions
Y = data[:, 0]            # Labels
X = data[:, 1:] / 255.0   # Normalized pixels
```

#### 3. **Normalization**
```python
# Scale pixel values from 0-255 to 0-1
X_normalized = X / 255.0
```

#### 4. **Train/Dev Split**
```python
# Split into training and development sets
data_dev = data[0:1000]     # First 1000 for validation
data_train = data[1000:]    # Remaining for training
```

## 🔍 Data Verification

### Check File Integrity

```python
import pandas as pd
import numpy as np

# Load and check training data
train_data = pd.read_csv("mnist_train.csv")
test_data = pd.read_csv("mnist_test.csv")

print(f"Training data shape: {train_data.shape}")  # Should be (60000, 785)
print(f"Test data shape: {test_data.shape}")       # Should be (10000, 785)

# Check label distribution
print("Training label distribution:")
print(train_data.iloc[:, 0].value_counts().sort_index())

# Check pixel value ranges
pixels = train_data.iloc[:, 1:].values
print(f"Pixel value range: {pixels.min()} to {pixels.max()}")
```

### Visualize Sample Images

```python
import matplotlib.pyplot as plt
import numpy as np

# Load data
data = pd.read_csv("mnist_test.csv")

# Display first 5 images
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i in range(5):
    # Get image data
    label = data.iloc[i, 0]
    pixels = data.iloc[i, 1:].values
    image = pixels.reshape(28, 28)
    
    # Display
    axes[i].imshow(image, cmap='gray')
    axes[i].set_title(f'Label: {label}')
    axes[i].axis('off')

plt.show()
```

## ⚠️ Common Issues and Solutions

### Issue 1: File Not Found
```
FileNotFoundError: [Errno 2] No such file or directory: 'mnist_test.csv'
```

**Solutions**:
- Ensure CSV files are in the same directory as your Python scripts
- Check file names are exactly `mnist_train.csv` and `mnist_test.csv`
- Verify files downloaded completely (check file sizes)

### Issue 2: Memory Error
```
MemoryError: Unable to allocate array
```

**Solutions**:
- Close other memory-intensive applications
- Use a subset of data for testing:
  ```python
  # Use only first 10,000 samples
  data = data.iloc[:10000]
  ```
- Use a machine with more RAM

### Issue 3: Slow Loading
**Solutions**:
- Use SSD storage instead of HDD
- Consider using Parquet format for faster loading:
  ```python
  # Convert to Parquet (one-time)
  df = pd.read_csv("mnist_train.csv")
  df.to_parquet("mnist_train.parquet")
  
  # Load Parquet (faster)
  df = pd.read_parquet("mnist_train.parquet")
  ```

### Issue 4: Wrong Data Format
If you get shape errors, verify the CSV structure:
```python
# Check first few rows
print(data.head())
print(f"Expected shape: (samples, 785), Got: {data.shape}")
```

## 📈 Dataset Alternatives

### 1. **Fashion-MNIST**
- Similar structure to MNIST
- 10 classes of clothing items
- Same 28×28 format
- More challenging than digits

### 2. **CIFAR-10**
- 32×32 color images
- 10 classes (planes, cars, etc.)
- More complex than MNIST
- Requires different preprocessing

### 3. **SVHN (Street View House Numbers)**
- Real-world digit images
- More challenging than MNIST
- Color images with varying backgrounds

## 🎯 Usage in Different Implementations

### NumPy Implementation
- Uses `mnist_test.csv` for both training and testing
- Splits the 10,000 samples: 1,000 for dev, 9,000 for training
- Manual data preprocessing

### PyTorch Implementation
- Same data split as NumPy
- Automatic tensor conversion
- GPU memory optimization

### TensorFlow Implementation
- Same data but with Keras preprocessing
- Automatic one-hot encoding
- Optimized data pipeline

## 📚 Learning Resources

### Understanding MNIST
- [Original MNIST Database](http://yann.lecun.com/exdb/mnist/)
- [Wikipedia: MNIST](https://en.wikipedia.org/wiki/MNIST_database)
- [Deep Learning Book Chapter 5](https://www.deeplearningbook.org/)

### Data Preprocessing
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [NumPy Array Manipulation](https://numpy.org/doc/stable/user/basics.html)
- [Image Processing with Python](https://scikit-image.org/docs/dev/user_guide.html)

---

**Ready to start training? Make sure your MNIST files are downloaded and in the right place! 🚀**
