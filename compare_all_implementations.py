import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

# Import original implementation functions
from artificial_neural_network_MNIST import *

# Import PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from artificial_neural_network_MNIST_pytorch import MNISTNet, train_pytorch_model, evaluate_pytorch_model

# Import TensorFlow
import tensorflow as tf
from tensorflow import keras
from artificial_neural_network_MNIST_tensorflow import create_tensorflow_model, train_tensorflow_model, evaluate_tensorflow_model

def run_comparison():
    """
    Compare all three implementations: NumPy, PyTorch, and TensorFlow
    """
    print("="*80)
    print("COMPREHENSIVE COMPARISON: NumPy vs PyTorch vs TensorFlow")
    print("="*80)
    
    # Load and prepare data (common for all implementations)
    print("Loading and preparing data...")
    data = pd.read_csv("mnist_test.csv")
    data = np.array(data)
    np.random.shuffle(data)
    
    # Split data
    data_dev = data[0:1000]
    data_train = data[1000:]
    
    # Prepare data for different frameworks
    results = {}
    
    # ===========================================
    # 1. ORIGINAL NUMPY IMPLEMENTATION
    # ===========================================
    print("\n" + "="*50)
    print("1. ORIGINAL NUMPY IMPLEMENTATION")
    print("="*50)
    
    start_time = time.time()
    
    # Prepare data for NumPy version
    Y_dev_np = data_dev[:, 0]
    X_dev_np = data_dev[:, 1:].T / 255.0
    Y_train_np = data_train[:, 0]
    X_train_np = data_train[:, 1:].T / 255.0
    
    Y_train_one_hot_np = one_hot(Y_train_np.astype(int))
    Y_dev_one_hot_np = one_hot(Y_dev_np.astype(int))
    
    # Train NumPy model
    W1_np, b1_np, W2_np, b2_np = gradient_descent(X_train_np, Y_train_one_hot_np, 0.1, 500)
    
    # Evaluate NumPy model
    _, _, _, A2_dev_np = forward_prop(W1_np, b1_np, W2_np, b2_np, X_dev_np)
    numpy_predictions = get_predictions(A2_dev_np)
    numpy_accuracy = get_accuracy(numpy_predictions, Y_dev_one_hot_np)
    
    numpy_time = time.time() - start_time
    
    results['numpy'] = {
        'accuracy': numpy_accuracy,
        'time': numpy_time,
        'framework': 'NumPy (Pure Python)',
        'parameters': 7850  # (10*784 + 10) + (10*10 + 10)
    }
    
    print(f"NumPy Implementation - Accuracy: {numpy_accuracy:.4f}, Time: {numpy_time:.2f}s")
    
    # ===========================================
    # 2. PYTORCH IMPLEMENTATION
    # ===========================================
    print("\n" + "="*50)
    print("2. PYTORCH IMPLEMENTATION")
    print("="*50)
    
    start_time = time.time()
    
    # Prepare data for PyTorch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train_torch = torch.FloatTensor(data_train[:, 1:] / 255.0).to(device)
    Y_train_torch = torch.LongTensor(data_train[:, 0]).to(device)
    X_dev_torch = torch.FloatTensor(data_dev[:, 1:] / 255.0).to(device)
    Y_dev_torch = torch.LongTensor(data_dev[:, 0]).to(device)
    
    # Train PyTorch model
    pytorch_model = MNISTNet().to(device)
    trained_pytorch_model = train_pytorch_model(
        pytorch_model, X_train_torch, Y_train_torch, 
        X_dev_torch, Y_dev_torch, epochs=500, learning_rate=0.1
    )
    
    # Evaluate PyTorch model
    pytorch_accuracy, _, _ = evaluate_pytorch_model(trained_pytorch_model, X_dev_torch, Y_dev_torch)
    pytorch_accuracy = pytorch_accuracy.cpu().numpy()
    
    pytorch_time = time.time() - start_time
    
    # Count PyTorch parameters
    pytorch_params = sum(p.numel() for p in trained_pytorch_model.parameters())
    
    results['pytorch'] = {
        'accuracy': pytorch_accuracy,
        'time': pytorch_time,
        'framework': f'PyTorch ({device})',
        'parameters': pytorch_params
    }
    
    print(f"PyTorch Implementation - Accuracy: {pytorch_accuracy:.4f}, Time: {pytorch_time:.2f}s")
    
    # ===========================================
    # 3. TENSORFLOW IMPLEMENTATION
    # ===========================================
    print("\n" + "="*50)
    print("3. TENSORFLOW IMPLEMENTATION")
    print("="*50)
    
    start_time = time.time()
    
    # Prepare data for TensorFlow
    X_train_tf = data_train[:, 1:].astype('float32') / 255.0
    Y_train_tf = keras.utils.to_categorical(data_train[:, 0], 10)
    X_dev_tf = data_dev[:, 1:].astype('float32') / 255.0
    Y_dev_tf = keras.utils.to_categorical(data_dev[:, 0], 10)
    
    # Train TensorFlow model
    tf_model = create_tensorflow_model()
    trained_tf_model, _ = train_tensorflow_model(
        tf_model, X_train_tf, Y_train_tf, 
        X_dev_tf, Y_dev_tf, epochs=500, learning_rate=0.1, batch_size=32
    )
    
    # Evaluate TensorFlow model
    tf_accuracy, _, _, _ = evaluate_tensorflow_model(trained_tf_model, X_dev_tf, Y_dev_tf)
    
    tensorflow_time = time.time() - start_time
    
    # Count TensorFlow parameters
    tf_params = trained_tf_model.count_params()
    
    results['tensorflow'] = {
        'accuracy': tf_accuracy,
        'time': tensorflow_time,
        'framework': 'TensorFlow/Keras',
        'parameters': tf_params
    }
    
    print(f"TensorFlow Implementation - Accuracy: {tf_accuracy:.4f}, Time: {tensorflow_time:.2f}s")
    
    # ===========================================
    # COMPARISON RESULTS
    # ===========================================
    print("\n" + "="*80)
    print("FINAL COMPARISON RESULTS")
    print("="*80)
    
    print(f"{'Framework':<20} {'Accuracy':<10} {'Time (s)':<10} {'Parameters':<12} {'Device'}")
    print("-" * 70)
    
    for name, result in results.items():
        print(f"{result['framework']:<20} {result['accuracy']:<10.4f} {result['time']:<10.2f} "
              f"{result['parameters']:<12} ")
    
    # Performance analysis
    print("\n" + "="*50)
    print("PERFORMANCE ANALYSIS")
    print("="*50)
    
    best_accuracy = max(results.values(), key=lambda x: x['accuracy'])
    fastest_training = min(results.values(), key=lambda x: x['time'])
    
    print(f"🏆 Best Accuracy: {best_accuracy['framework']} ({best_accuracy['accuracy']:.4f})")
    print(f"⚡ Fastest Training: {fastest_training['framework']} ({fastest_training['time']:.2f}s)")
    
    # Accuracy comparison
    numpy_acc = results['numpy']['accuracy']
    pytorch_acc = results['pytorch']['accuracy']
    tf_acc = results['tensorflow']['accuracy']
    
    print(f"\n📊 Accuracy Differences:")
    print(f"   PyTorch vs NumPy: {(pytorch_acc - numpy_acc)*100:+.2f}%")
    print(f"   TensorFlow vs NumPy: {(tf_acc - numpy_acc)*100:+.2f}%")
    print(f"   TensorFlow vs PyTorch: {(tf_acc - pytorch_acc)*100:+.2f}%")
    
    # Speed comparison
    numpy_time = results['numpy']['time']
    pytorch_time = results['pytorch']['time']
    tf_time = results['tensorflow']['time']
    
    print(f"\n⏱️ Speed Improvements:")
    print(f"   PyTorch vs NumPy: {numpy_time/pytorch_time:.1f}x {'faster' if pytorch_time < numpy_time else 'slower'}")
    print(f"   TensorFlow vs NumPy: {numpy_time/tf_time:.1f}x {'faster' if tf_time < numpy_time else 'slower'}")
    
    # Create visualization
    create_comparison_plot(results)
    
    return results


def create_comparison_plot(results):
    """
    Create visualization comparing all implementations
    """
    frameworks = list(results.keys())
    accuracies = [results[fw]['accuracy'] for fw in frameworks]
    times = [results[fw]['time'] for fw in frameworks]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy comparison
    bars1 = ax1.bar(frameworks, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Training time comparison
    bars2 = ax2.bar(frameworks, times, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax2.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Time (seconds)')
    
    # Add value labels on bars
    for bar, time_val in zip(bars2, times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(times)*0.01,
                f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Create parameter comparison
    plt.figure(figsize=(10, 6))
    params = [results[fw]['parameters'] for fw in frameworks]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    bars = plt.bar(frameworks, params, color=colors)
    plt.title('Model Parameters Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Parameters')
    
    for bar, param in zip(bars, params):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(params)*0.01,
                f'{param:,}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    results = run_comparison()
