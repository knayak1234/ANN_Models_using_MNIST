# 🎯 Usage Examples and Tutorials

This guide provides practical examples of how to use the neural network implementations and customize them for different scenarios.

## 🚀 Basic Usage Examples

### 1. Running Individual Implementations

#### NumPy Implementation
```bash
# Basic run
python artificial_neural_network_MNIST.py

# Expected output:
# Original data shape: (10000, 785)
# Data dimensions: 10000 samples, 785 features
# ...training progress...
# Development set accuracy: 0.8543
```

#### PyTorch Implementation
```bash
# Run with GPU support (if available)
python artificial_neural_network_MNIST_pytorch.py

# Expected output:
# Using device: cuda
# Training data shape: torch.Size([9000, 784])
# ...training progress...
# Final PyTorch Development Set Accuracy: 0.9120
```

#### TensorFlow Implementation
```bash
# Run with advanced features
python artificial_neural_network_MNIST_tensorflow.py

# Expected output:
# TensorFlow version: 2.16.2
# Model Summary:
# ...training progress...
# Advanced TensorFlow Model Results: Accuracy: 0.9240
```

### 2. Performance Comparison

```bash
# Compare all three implementations
python compare_all_implementations.py

# Expected output:
# COMPREHENSIVE COMPARISON: NumPy vs PyTorch vs TensorFlow
# ================================================
# Framework          Accuracy   Time (s)   Parameters
# NumPy (Pure Python) 0.8543    89.23      7850
# PyTorch (CPU)       0.9120    34.56      7850  
# TensorFlow/Keras    0.9240    28.14      7850
```

## 🔧 Customization Examples

### 1. Modifying Network Architecture

#### Increase Hidden Layer Size
```python
# In config.py
HIDDEN_LAYER_SIZE = 64  # Default: 10

# Or modify directly in the code:
# NumPy version (lines 34, 36)
W1 = np.random.rand(64, 784) - 0.5  # 64 instead of 10
W2 = np.random.rand(10, 64) - 0.5   # 64 instead of 10

# PyTorch version
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 64)  # 64 instead of 10
        self.fc2 = nn.Linear(64, 10)   # 64 instead of 10

# TensorFlow version
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(784,)),  # 64 instead of 10
    layers.Dense(10, activation='softmax')
])
```

#### Add More Layers
```python
# PyTorch: Multi-layer network
class DeepMNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 10)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# TensorFlow: Deep network with regularization
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

### 2. Adjusting Training Parameters

#### Learning Rate Experiments
```python
# Test different learning rates
learning_rates = [0.001, 0.01, 0.1, 0.5, 1.0]
results = {}

for lr in learning_rates:
    print(f"Testing learning rate: {lr}")
    
    # NumPy version
    W1, b1, W2, b2 = gradient_descent(X_train_norm, Y_train_one_hot, lr, 200)
    accuracy = evaluate_model(W1, b1, W2, b2, X_dev_norm, Y_dev_one_hot)
    results[lr] = accuracy
    
    print(f"Accuracy with LR {lr}: {accuracy:.4f}")

# Find best learning rate
best_lr = max(results, key=results.get)
print(f"Best learning rate: {best_lr} (Accuracy: {results[best_lr]:.4f})")
```

#### Training Duration Analysis
```python
# Test different iteration counts
iterations = [100, 200, 500, 1000, 2000]
accuracies = []

for iters in iterations:
    W1, b1, W2, b2 = gradient_descent(X_train_norm, Y_train_one_hot, 0.1, iters)
    accuracy = get_final_accuracy(W1, b1, W2, b2, X_dev_norm, Y_dev_one_hot)
    accuracies.append(accuracy)
    print(f"Iterations: {iters:4d} | Accuracy: {accuracy:.4f}")

# Plot learning curve
import matplotlib.pyplot as plt
plt.plot(iterations, accuracies, 'bo-')
plt.xlabel('Training Iterations')
plt.ylabel('Accuracy')
plt.title('Learning Curve: Iterations vs Accuracy')
plt.grid(True)
plt.show()
```

### 3. Data Augmentation Examples

#### Simple Data Augmentation (NumPy)
```python
def augment_data(X, Y, noise_factor=0.1):
    """Add noise to training data for better generalization"""
    # Add Gaussian noise
    noise = np.random.normal(0, noise_factor, X.shape)
    X_augmented = np.clip(X + noise, 0, 1)
    
    # Duplicate labels
    Y_augmented = Y
    
    return X_augmented, Y_augmented

# Usage
X_train_aug, Y_train_aug = augment_data(X_train_norm, Y_train_one_hot)
W1, b1, W2, b2 = gradient_descent(X_train_aug, Y_train_aug, 0.1, 500)
```

#### Advanced Augmentation (TensorFlow)
```python
# Data augmentation with tf.keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create data generator
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest'
)

# Reshape data for image augmentation
X_train_reshaped = X_train.reshape(-1, 28, 28, 1)

# Fit and transform
datagen.fit(X_train_reshaped)
augmented_data = datagen.flow(X_train_reshaped, Y_train_categorical, batch_size=32)

# Train with augmented data
model.fit(augmented_data, epochs=100, validation_data=(X_dev_reshaped, Y_dev_categorical))
```

## 📊 Analysis and Visualization Examples

### 1. Training Progress Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_training_history(accuracies, losses=None):
    """Plot training progress"""
    epochs = range(len(accuracies))
    
    if losses:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy plot
        ax1.plot(epochs, accuracies, 'b-', label='Training Accuracy')
        ax1.set_title('Training Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.grid(True)
        
        # Loss plot
        ax2.plot(epochs, losses, 'r-', label='Training Loss')
        ax2.set_title('Training Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.grid(True)
    else:
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, accuracies, 'b-', linewidth=2)
        plt.title('Training Accuracy Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Usage with stored training history
training_accuracies = []  # Store accuracies during training
plot_training_history(training_accuracies)
```

### 2. Prediction Analysis

```python
def analyze_predictions(model, X_test, Y_test, num_samples=10):
    """Analyze model predictions with confidence scores"""
    
    # Get predictions (adjust based on framework)
    if framework == 'numpy':
        _, _, _, A2 = forward_prop(W1, b1, W2, b2, X_test)
        predictions = np.argmax(A2, axis=0)
        confidences = np.max(A2, axis=0)
    elif framework == 'pytorch':
        with torch.no_grad():
            outputs = model(X_test)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            confidences = torch.softmax(outputs, dim=1).max(dim=1)[0].cpu().numpy()
    
    # Analyze results
    correct_predictions = predictions == Y_test
    accuracy = np.mean(correct_predictions)
    
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Confident Correct Predictions (>0.9): {np.sum((confidences > 0.9) & correct_predictions)}")
    print(f"Low Confidence Errors (<0.6): {np.sum((confidences < 0.6) & ~correct_predictions)}")
    
    # Show some examples
    for i in range(min(num_samples, len(predictions))):
        status = "✓" if correct_predictions[i] else "✗"
        print(f"Sample {i}: Pred={predictions[i]}, True={Y_test[i]}, "
              f"Confidence={confidences[i]:.3f} {status}")

# Usage
analyze_predictions(model, X_dev_tensor, Y_dev_tensor)
```

### 3. Error Analysis

```python
def confusion_matrix_analysis(y_true, y_pred):
    """Create and analyze confusion matrix"""
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    # Find most confused classes
    np.fill_diagonal(cm, 0)  # Remove correct predictions
    most_confused = np.unravel_index(cm.argmax(), cm.shape)
    print(f"\nMost confused: {most_confused[0]} predicted as {most_confused[1]} "
          f"({cm[most_confused]} times)")

# Usage
confusion_matrix_analysis(Y_dev, dev_predictions)
```

## 🔬 Advanced Experiments

### 1. Architecture Comparison

```python
def compare_architectures():
    """Compare different network architectures"""
    
    architectures = {
        'Small': [784, 5, 10],
        'Original': [784, 10, 10], 
        'Medium': [784, 32, 10],
        'Large': [784, 64, 10],
        'Deep': [784, 32, 16, 10]
    }
    
    results = {}
    
    for name, layers in architectures.items():
        print(f"Testing {name} architecture: {layers}")
        
        # Create and train model (PyTorch example)
        model = create_model(layers)
        train_time = time.time()
        trained_model = train_model(model, X_train, Y_train, epochs=200)
        train_time = time.time() - train_time
        
        # Evaluate
        accuracy = evaluate_model(trained_model, X_dev, Y_dev)
        params = count_parameters(trained_model)
        
        results[name] = {
            'accuracy': accuracy,
            'parameters': params,
            'train_time': train_time
        }
        
        print(f"  Accuracy: {accuracy:.4f}, Parameters: {params}, Time: {train_time:.2f}s")
    
    return results

# Run comparison
arch_results = compare_architectures()
```

### 2. Optimizer Comparison (PyTorch)

```python
def compare_optimizers():
    """Compare different optimization algorithms"""
    
    optimizers = {
        'SGD': lambda params: optim.SGD(params, lr=0.01),
        'Adam': lambda params: optim.Adam(params, lr=0.001),
        'RMSprop': lambda params: optim.RMSprop(params, lr=0.001),
        'AdaGrad': lambda params: optim.Adagrad(params, lr=0.01)
    }
    
    results = {}
    
    for name, opt_fn in optimizers.items():
        print(f"Testing {name} optimizer...")
        
        model = MNISTNet()
        optimizer = opt_fn(model.parameters())
        
        # Train with different optimizer
        accuracy = train_and_evaluate(model, optimizer, X_train, Y_train, X_dev, Y_dev)
        results[name] = accuracy
        
        print(f"  {name} Accuracy: {accuracy:.4f}")
    
    return results

# Run optimizer comparison
opt_results = compare_optimizers()
```

### 3. Hyperparameter Grid Search

```python
def grid_search():
    """Perform hyperparameter grid search"""
    
    param_grid = {
        'learning_rate': [0.001, 0.01, 0.1],
        'hidden_size': [16, 32, 64],
        'batch_size': [16, 32, 64]
    }
    
    best_score = 0
    best_params = {}
    results = []
    
    for lr in param_grid['learning_rate']:
        for hidden in param_grid['hidden_size']:
            for batch in param_grid['batch_size']:
                
                print(f"Testing: LR={lr}, Hidden={hidden}, Batch={batch}")
                
                # Train model with these parameters
                model = create_model(hidden_size=hidden)
                accuracy = train_model(model, lr=lr, batch_size=batch)
                
                results.append({
                    'lr': lr, 'hidden': hidden, 'batch': batch, 
                    'accuracy': accuracy
                })
                
                if accuracy > best_score:
                    best_score = accuracy
                    best_params = {'lr': lr, 'hidden': hidden, 'batch': batch}
                
                print(f"  Accuracy: {accuracy:.4f}")
    
    print(f"\nBest parameters: {best_params}")
    print(f"Best accuracy: {best_score:.4f}")
    
    return results, best_params

# Run grid search
search_results, best_config = grid_search()
```

## 🎯 Real-World Applications

### 1. Deploy Model as Web Service (Flask + TensorFlow)

```python
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model('mnist_tensorflow_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    # Get image from request
    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read()))
    
    # Preprocess image
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image_array = np.array(image) / 255.0  # Normalize
    image_array = image_array.reshape(1, 784)  # Flatten
    
    # Make prediction
    prediction = model.predict(image_array)
    digit = np.argmax(prediction)
    confidence = float(np.max(prediction))
    
    return jsonify({
        'digit': int(digit),
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(debug=True)
```

### 2. Real-time Digit Recognition (OpenCV)

```python
import cv2
import numpy as np
import torch

# Load trained PyTorch model
model = MNISTNet()
model.load_state_dict(torch.load('mnist_pytorch_model.pth'))
model.eval()

def preprocess_frame(frame):
    """Preprocess camera frame for digit recognition"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    normalized = resized.astype(np.float32) / 255.0
    return normalized.reshape(1, 784)

# Start camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Get region of interest (center square)
    h, w = frame.shape[:2]
    size = min(h, w) // 3
    center_x, center_y = w // 2, h // 2
    roi = frame[center_y-size//2:center_y+size//2, 
                center_x-size//2:center_x+size//2]
    
    # Preprocess and predict
    processed = preprocess_frame(roi)
    with torch.no_grad():
        prediction = model(torch.FloatTensor(processed))
        digit = torch.argmax(prediction).item()
        confidence = torch.softmax(prediction, dim=1).max().item()
    
    # Display results
    cv2.rectangle(frame, (center_x-size//2, center_y-size//2), 
                  (center_x+size//2, center_y+size//2), (0, 255, 0), 2)
    cv2.putText(frame, f'Digit: {digit} ({confidence:.2f})', 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Digit Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 📚 Learning Exercises

### Exercise 1: Implement Different Activation Functions
Try replacing ReLU with other activation functions and compare results:
- Sigmoid
- Tanh
- Leaky ReLU
- Swish

### Exercise 2: Add Regularization
Implement different regularization techniques:
- L1/L2 weight decay
- Dropout
- Batch normalization
- Early stopping

### Exercise 3: Multi-class to Binary Classification
Modify the network to classify odd vs even digits (binary classification).

### Exercise 4: Transfer Learning
Use a pre-trained model and fine-tune it for MNIST classification.

---

**These examples should give you a solid foundation for experimenting with neural networks! 🚀**
