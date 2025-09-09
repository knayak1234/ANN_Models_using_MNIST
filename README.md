# 🧠 MNIST Neural Network: NumPy → PyTorch → TensorFlow

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26%2B-orange.svg)](https://numpy.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-red.svg)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Learn neural networks from the ground up!** This educational project implements the same MNIST digit classification network in **three different ways**: pure NumPy, PyTorch, and TensorFlow, showing the evolution from fundamentals to modern deep learning.

## 🎯 Project Overview

This repository contains **three implementations** of a neural network for handwritten digit classification:

1. **📐 NumPy Implementation**: Built from scratch to understand the fundamentals
2. **🔥 PyTorch Implementation**: Modern deep learning with automatic differentiation
3. **🧠 TensorFlow Implementation**: High-level API with production-ready features
4. **📊 Performance Comparison**: Side-by-side analysis of all three approaches

### 🏗️ Network Architecture

All implementations use the same neural network structure:
- **Input Layer**: 784 neurons (28×28 pixel images)
- **Hidden Layer**: 10 neurons with ReLU activation
- **Output Layer**: 10 neurons with Softmax (digits 0-9)

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/mnist-neural-network-comparison.git
cd mnist-neural-network-comparison
```

### 2. Install Dependencies
```bash
# Option 1: Use the enhanced installer (recommended)
chmod +x install_fixed.sh
./install_fixed.sh

# Option 2: Manual installation
pip install -r requirements.txt
```

### 3. Download MNIST Dataset
Download the MNIST CSV files from [Kaggle](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv):
- `mnist_train.csv` (~109 MB)
- `mnist_test.csv` (~18 MB)

Place both files in the project directory.

### 4. Run the Implementations

```bash
# Original NumPy implementation
python artificial_neural_network_MNIST.py

# PyTorch implementation
python artificial_neural_network_MNIST_pytorch.py

# TensorFlow implementation
python artificial_neural_network_MNIST_tensorflow.py

# Compare all three implementations
python compare_all_implementations.py
```

## 📁 Project Structure

```
mnist-neural-network-comparison/
├── 📄 README.md                                    # This file
├── 📊 PROJECT_SUMMARY.md                           # Quick project overview
├── 📚 README_MODERN_IMPLEMENTATIONS.md             # Detailed technical comparison
├── 🛠️ setup_guide.md                               # Comprehensive setup instructions
│
├── 🐍 Implementation Files
│   ├── artificial_neural_network_MNIST.py          # NumPy implementation
│   ├── artificial_neural_network_MNIST_pytorch.py  # PyTorch implementation
│   ├── artificial_neural_network_MNIST_tensorflow.py # TensorFlow implementation
│   └── compare_all_implementations.py              # Performance comparison
│
├── ⚙️ Configuration & Setup
│   ├── requirements.txt                            # Python dependencies
│   ├── config.py                                   # Configurable parameters
│   ├── run.py                                      # Simple runner script
│   └── install_fixed.sh                           # Enhanced installer
│
├── 📊 Data Files (download separately)
│   ├── mnist_train.csv                             # Training data (60K samples)
│   └── mnist_test.csv                              # Test data (10K samples)
│
└── 🎯 Model Outputs
    ├── mnist_pytorch_model.pth                     # Saved PyTorch model
    ├── mnist_tensorflow_simple_model.h5            # Simple TensorFlow model
    └── mnist_tensorflow_advanced_model.h5          # Advanced TensorFlow model
```

## 🔍 Implementation Comparison

| Feature | NumPy | PyTorch | TensorFlow |
|---------|-------|---------|------------|
| **Learning Curve** | 📈 Educational | 📊 Moderate | 📉 Easy |
| **Performance** | 🐌 85-90% | 🚀 87-92% | ⚡ 88-93% |
| **Training Speed** | 🕒 60-120s | 🕐 30-60s | ⏰ 20-40s |
| **GPU Support** | ❌ No | ✅ Yes | ✅ Yes |
| **Automatic Gradients** | ❌ Manual | ✅ Yes | ✅ Yes |
| **Production Ready** | ❌ Educational | ✅ Research | ✅ Production |
| **Code Complexity** | 📝 ~200 lines | 📄 ~150 lines | 📃 ~100 lines |

## 🧮 Key Learning Differences

### 1. **Gradient Computation**

**NumPy** (Manual Implementation):
```python
def back_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    m = Y.shape[1]
    dZ2 = A2 - Y
    dW2 = 1 / m * dZ2 @ A1.T
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    # ... manual gradient calculations
```

**PyTorch** (Automatic Differentiation):
```python
loss = criterion(outputs, targets)
loss.backward()  # Automatically computes all gradients!
optimizer.step()
```

**TensorFlow** (High-level API):
```python
model.fit(X_train, Y_train, epochs=500)  # Everything handled automatically!
```

### 2. **Model Definition**

**NumPy**:
```python
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1 @ X + b1
    A1 = ReLU(Z1)
    Z2 = W2 @ A1 + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2
```

**PyTorch**:
```python
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 10)
        self.fc2 = nn.Linear(10, 10)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

**TensorFlow**:
```python
model = keras.Sequential([
    layers.Dense(10, activation='relu', input_shape=(784,)),
    layers.Dense(10, activation='softmax')
])
```

## 📊 Performance Results

### Expected Accuracy Comparison
- **NumPy**: 85-90% (baseline implementation)
- **PyTorch**: 87-92% (+2-5% improvement from better optimization)
- **TensorFlow**: 88-93% (+3-8% improvement from advanced features)

### Speed Improvements
- **PyTorch**: 3-5x faster than NumPy
- **TensorFlow**: 4-6x faster than NumPy
- **GPU Acceleration**: Up to 10x faster on compatible hardware

## 🛠️ Advanced Features

### PyTorch Implementation
- ✅ GPU acceleration support
- ✅ Batch processing with DataLoader
- ✅ Model saving/loading
- ✅ Custom training loops
- ✅ Automatic mixed precision ready

### TensorFlow Implementation
- ✅ Two model architectures (simple + advanced)
- ✅ Training callbacks (early stopping, learning rate scheduling)
- ✅ Training history visualization
- ✅ TensorBoard integration ready
- ✅ Model export for deployment

### Comparison Script
- ✅ Side-by-side performance analysis
- ✅ Timing benchmarks
- ✅ Accuracy comparisons
- ✅ Visual performance charts
- ✅ Parameter counting

## 🎓 Educational Value

### Choose **NumPy** to Learn:
- How neural networks actually work
- Matrix operations and linear algebra
- Gradient descent mathematics
- Backpropagation algorithm
- Complete control over every operation

### Choose **PyTorch** to Learn:
- Modern deep learning workflows
- Automatic differentiation
- GPU acceleration
- Research-oriented development
- Dynamic computation graphs

### Choose **TensorFlow** to Learn:
- Production machine learning
- High-level APIs (Keras)
- Model deployment
- Industry-standard practices
- Large-scale ML systems

## 🔧 Customization

### Modify Network Architecture
```python
# In config.py
HIDDEN_LAYER_SIZE = 64  # Increase hidden layer size
LEARNING_RATE = 0.01    # Adjust learning rate
ITERATIONS = 1000       # More training iterations
```

### Advanced Experiments
- Add more hidden layers
- Try different activation functions
- Implement different optimizers
- Add regularization techniques
- Experiment with data augmentation

## 📈 Performance Optimization

### For Better Accuracy:
1. Increase hidden layer size (10 → 64 → 128)
2. Add more layers
3. Use better optimizers (Adam, RMSprop)
4. Add dropout regularization
5. Implement data augmentation

### For Faster Training:
1. Use GPU acceleration (PyTorch/TensorFlow)
2. Increase batch size
3. Use mixed precision training
4. Optimize data loading
5. Use compiled models (TensorFlow)

## 🐛 Troubleshooting

### Common Issues and Solutions

#### Installation Problems
- **Python 3.12 compatibility**: Use `install_fixed.sh` script
- **GPU not detected**: Install CUDA-compatible versions
- **Memory errors**: Reduce batch size or model size

#### Runtime Errors
- **File not found**: Download MNIST CSV files from Kaggle
- **Shape mismatches**: Check data preprocessing steps
- **Low accuracy**: Increase training iterations or learning rate

#### Performance Issues
- **Slow training**: Enable GPU acceleration or reduce model size
- **Poor convergence**: Adjust learning rate or initialization

## 📚 Resources for Further Learning

### Online Courses
- [Deep Learning Specialization (Coursera)](https://www.coursera.org/specializations/deep-learning)
- [CS231n: CNN for Visual Recognition (Stanford)](http://cs231n.stanford.edu/)
- [Fast.ai Practical Deep Learning](https://course.fast.ai/)

### Books
- [Deep Learning by Ian Goodfellow](https://www.deeplearningbook.org/)
- [Neural Networks and Deep Learning (online)](http://neuralnetworksanddeeplearning.com/)
- [Hands-On Machine Learning by Aurélien Géron](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)

### Documentation
- [PyTorch Documentation](https://pytorch.org/docs/)
- [TensorFlow Documentation](https://www.tensorflow.org/guide)
- [NumPy Documentation](https://numpy.org/doc/)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Contribution Ideas
- Add more frameworks (JAX, MLX)
- Implement different architectures (CNN, RNN)
- Add visualization tools
- Improve documentation
- Add unit tests
- Performance optimizations

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MNIST Dataset**: Yann LeCun, Corinna Cortes, Christopher J.C. Burges
- **Educational Inspiration**: Andrew Ng's Machine Learning Course
- **Modern Framework Examples**: PyTorch and TensorFlow communities
- **Kaggle**: For providing the MNIST CSV format

## 📞 Contact

- **Author**: Your Name
- **Email**: your.email@example.com
- **GitHub**: [@yourusername](https://github.com/yourusername)
- **LinkedIn**: [Your LinkedIn](https://linkedin.com/in/yourprofile)

---

### ⭐ If this project helped you learn neural networks, please give it a star! ⭐

**Happy Learning! 🚀**