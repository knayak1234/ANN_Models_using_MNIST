# Artificial Neural Network for MNIST Digit Classification

## Overview

This project implements a **two-layer artificial neural network from scratch** to classify handwritten digits from the MNIST dataset. The implementation uses only NumPy for mathematical operations, providing a clear understanding of the fundamental concepts behind neural networks without relying on high-level frameworks.

## 🎯 Project Goals

- Build a neural network from scratch using only NumPy
- Understand forward propagation, backpropagation, and gradient descent
- Classify handwritten digits (0-9) from the MNIST dataset
- Achieve good accuracy on the test set

## 🏗️ Architecture

### Network Structure
- **Input Layer**: 784 neurons (28×28 pixel images flattened)
- **Hidden Layer**: 10 neurons with ReLU activation
- **Output Layer**: 10 neurons with Softmax activation (one for each digit 0-9)

### Key Components
1. **Forward Propagation**: Computes predictions through the network
2. **Backpropagation**: Calculates gradients for parameter updates
3. **Gradient Descent**: Updates weights and biases to minimize loss
4. **Activation Functions**:
   - ReLU for hidden layer
   - Softmax for output layer

## 📁 Project Structure

```
first_ANN_model/
├── artificial_neural_network_MNIST.py  # Main implementation
├── mnist_train.csv                     # Training data (not included)
├── mnist_test.csv                      # Test data (not included)
├── requirements.txt                    # Python dependencies
├── README.md                          # This file
└── setup_guide.md                     # Detailed setup instructions
```

## 🚀 Quick Start

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone or download the project**
   ```bash
   cd /path/to/first_ANN_model
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download MNIST dataset**
   - Download `mnist_train.csv` and `mnist_test.csv` from [Kaggle MNIST dataset](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
   - Place both files in the project directory

4. **Run the program**
   ```bash
   python artificial_neural_network_MNIST.py
   ```

## 📊 Expected Output

When you run the program, you'll see:

1. **Data Loading Information**:
   ```
   Original data shape: (10000, 785)
   Data dimensions: 10000 samples, 785 features
   Dev set - X shape: (784, 1000), Y shape: (1000,)
   Train set - X shape: (784, 9000), Y shape: (9000,)
   ```

2. **Training Progress** (every 10 iterations):
   ```
   Iteration: 0
   Accuracy: 0.1234
   Iteration: 10
   Accuracy: 0.2345
   ...
   ```

3. **Final Results**:
   ```
   Development set accuracy: 0.8500
   Sample predictions:
   Prediction:  [7]
   Label:  7
   [Image display]
   ```

## 🔧 Code Explanation

### Main Functions

#### Data Preprocessing
- **Data Loading**: Loads MNIST CSV files using pandas
- **Data Splitting**: Splits into training (9000 samples) and dev (1000 samples) sets
- **Normalization**: Scales pixel values from 0-255 to 0-1
- **One-hot Encoding**: Converts labels to one-hot vectors

#### Neural Network Functions

```python
# Initialize parameters with random values
def init_params():
    W1 = np.random.rand(10, 784) - 0.5  # Hidden layer weights
    b1 = np.random.rand(10, 1) - 0.5    # Hidden layer biases
    W2 = np.random.rand(10, 10) - 0.5   # Output layer weights
    b2 = np.random.rand(10, 1) - 0.5    # Output layer biases
    return W1, b1, W2, b2

# Forward propagation
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1 @ X + b1          # Linear transformation (hidden layer)
    A1 = ReLU(Z1)             # ReLU activation
    Z2 = W2 @ A1 + b2         # Linear transformation (output layer)
    A2 = softmax(Z2)          # Softmax activation
    return Z1, A1, Z2, A2

# Backpropagation
def back_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    # Calculate gradients for all parameters
    # Returns dW1, db1, dW2, db2
```

#### Training Process
1. **Initialize** random weights and biases
2. **Forward pass** to get predictions
3. **Calculate loss** using cross-entropy
4. **Backward pass** to compute gradients
5. **Update parameters** using gradient descent
6. **Repeat** for specified iterations

## 📈 Performance

- **Training Set**: ~9000 samples
- **Dev Set**: ~1000 samples
- **Expected Accuracy**: 85-90% on development set
- **Training Time**: ~2-3 minutes for 500 iterations

## 🔬 Technical Details

### Activation Functions
- **ReLU**: `f(x) = max(0, x)` - Used in hidden layer
- **Softmax**: `f(x_i) = e^(x_i) / Σe^(x_j)` - Used in output layer

### Loss Function
- **Cross-entropy loss** for multi-class classification

### Optimization
- **Gradient Descent** with learning rate α = 0.1
- **Batch processing** for efficiency

## 🎨 Visualization

The program displays sample predictions with matplotlib, showing:
- The actual handwritten digit image
- The model's prediction
- The true label

## 🛠️ Customization

You can modify various parameters:

```python
# Training parameters
learning_rate = 0.1      # Line 111
iterations = 500         # Line 111

# Network architecture
hidden_units = 10        # Lines 34, 36

# Data split
dev_samples = 1000       # Line 18
```

## 📚 Learning Objectives

This implementation helps you understand:

1. **Matrix Operations**: How neural networks use linear algebra
2. **Gradient Descent**: How networks learn from data
3. **Backpropagation**: How gradients flow through layers
4. **Activation Functions**: How nonlinearity enables learning
5. **Classification**: How to build multi-class classifiers

## 🐛 Troubleshooting

### Common Issues

1. **File Not Found Error**:
   - Ensure MNIST CSV files are in the project directory
   - Check file paths in line 6 of the code

2. **Memory Error**:
   - Reduce batch size or use smaller dataset
   - Close other memory-intensive applications

3. **Low Accuracy**:
   - Increase number of iterations
   - Adjust learning rate
   - Check data preprocessing

4. **Import Errors**:
   - Install requirements: `pip install -r requirements.txt`
   - Check Python version compatibility

## 📖 Further Reading

- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
- [Deep Learning Specialization - Coursera](https://www.coursera.org/specializations/deep-learning)
- [CS231n: Convolutional Neural Networks](http://cs231n.stanford.edu/)

## 📄 License

This project is for educational purposes. Feel free to use and modify for learning.

## 👥 Contributing

This is an educational project. Suggestions and improvements are welcome!

---

**Happy Learning! 🚀**
