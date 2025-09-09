import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, metrics
from matplotlib import pyplot as plt
import time

# Set up TensorFlow configuration
print("TensorFlow version:", tf.__version__)
print("GPU Available: ", tf.config.list_physical_devices('GPU'))

# Load the data
print("Loading MNIST data...")
data = pd.read_csv("mnist_test.csv")
print(f"Original data shape: {data.shape}")

# Convert to numpy array
data = np.array(data)
m, n = data.shape
print(f"Data dimensions: {m} samples, {n} features")

# Shuffle the data BEFORE splitting
np.random.shuffle(data)

# Split into dev and train sets
data_dev = data[0:1000]  # First 1000 samples for development
data_train = data[1000:m]  # Remaining samples for training

# Extract features and labels for dev set
Y_dev = data_dev[:, 0]  # First column is labels
X_dev = data_dev[:, 1:n]  # Remaining columns are pixel features

# Extract features and labels for train set  
Y_train = data_train[:, 0]  # First column is labels
X_train = data_train[:, 1:n]  # Remaining columns are pixel features

print(f"Dev set - X shape: {X_dev.shape}, Y shape: {Y_dev.shape}")
print(f"Train set - X shape: {X_train.shape}, Y shape: {Y_train.shape}")

# Normalize the input data
X_train_norm = X_train.astype('float32') / 255.0
X_dev_norm = X_dev.astype('float32') / 255.0

# Convert labels to categorical (one-hot encoding)
Y_train_categorical = keras.utils.to_categorical(Y_train, 10)
Y_dev_categorical = keras.utils.to_categorical(Y_dev, 10)

print(f"Training data shape: {X_train_norm.shape}")
print(f"Training labels shape: {Y_train_categorical.shape}")


def create_tensorflow_model(input_size=784, hidden_size=10, output_size=10):
    """
    Create TensorFlow/Keras model for MNIST classification
    Architecture: Input(784) -> Hidden(10) -> Output(10)
    """
    model = keras.Sequential([
        layers.Dense(hidden_size, activation='relu', input_shape=(input_size,), name='hidden_layer'),
        layers.Dense(output_size, activation='softmax', name='output_layer')
    ])
    
    return model


def create_advanced_tensorflow_model(input_size=784, hidden_sizes=[128, 64], output_size=10, dropout_rate=0.2):
    """
    Create an advanced TensorFlow model with multiple layers and regularization
    """
    model = keras.Sequential([
        layers.Dense(hidden_sizes[0], activation='relu', input_shape=(input_size,), name='hidden_layer_1'),
        layers.Dropout(dropout_rate, name='dropout_1'),
        layers.Dense(hidden_sizes[1], activation='relu', name='hidden_layer_2'),
        layers.Dropout(dropout_rate, name='dropout_2'),
        layers.Dense(output_size, activation='softmax', name='output_layer')
    ])
    
    return model


def train_tensorflow_model(model, X_train, Y_train, X_val, Y_val, epochs=500, learning_rate=0.1, batch_size=32):
    """
    Train the TensorFlow model with modern techniques
    """
    # Compile the model
    model.compile(
        optimizer=optimizers.SGD(learning_rate=learning_rate),
        loss=losses.CategoricalCrossentropy(),
        metrics=[metrics.CategoricalAccuracy()]
    )
    
    print(f"\nModel Summary:")
    model.summary()
    
    # Create callbacks for better training
    callbacks = [
        keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=20, min_lr=0.001)
    ]
    
    print("\nStarting TensorFlow training...")
    start_time = time.time()
    
    # Train the model
    history = model.fit(
        X_train, Y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, Y_val),
        callbacks=callbacks,
        verbose=0  # We'll print our own progress
    )
    
    training_time = time.time() - start_time
    print(f"\nTensorFlow training completed in {training_time:.2f} seconds")
    
    return model, history


def evaluate_tensorflow_model(model, X_test, Y_test):
    """
    Evaluate the trained TensorFlow model
    """
    # Get predictions
    predictions_prob = model.predict(X_test, verbose=0)
    predictions = np.argmax(predictions_prob, axis=1)
    true_labels = np.argmax(Y_test, axis=1)
    
    # Calculate accuracy
    accuracy = np.mean(predictions == true_labels)
    
    # Get loss
    loss = model.evaluate(X_test, Y_test, verbose=0)[0]
    
    return accuracy, predictions, predictions_prob, loss


def visualize_tensorflow_predictions(model, X_test, Y_test, num_samples=5):
    """
    Visualize sample predictions from TensorFlow model
    """
    predictions_prob = model.predict(X_test[:num_samples], verbose=0)
    predictions = np.argmax(predictions_prob, axis=1)
    true_labels = np.argmax(Y_test[:num_samples], axis=1)
    
    print("\nTensorFlow Sample Predictions:")
    for i in range(num_samples):
        confidence = predictions_prob[i][predictions[i]]
        print(f"Sample {i+1}: Predicted={predictions[i]}, Actual={true_labels[i]}, "
              f"Confidence={confidence:.3f}")
        
        # Display image
        image = X_test[i].reshape(28, 28) * 255
        plt.figure(figsize=(4, 4))
        plt.gray()
        plt.imshow(image, interpolation='nearest')
        plt.title(f"TensorFlow: Pred={predictions[i]}, True={true_labels[i]}")
        plt.axis('off')
        plt.show()


def plot_training_history(history):
    """
    Plot training history
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot training & validation accuracy
    ax1.plot(history.history['categorical_accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_categorical_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot training & validation loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


# Simple TensorFlow model (matching original architecture)
print("\n" + "="*50)
print("TENSORFLOW IMPLEMENTATION - SIMPLE MODEL")
print("="*50)

simple_tf_model = create_tensorflow_model()
simple_tf_model, simple_history = train_tensorflow_model(
    simple_tf_model, X_train_norm, Y_train_categorical, 
    X_dev_norm, Y_dev_categorical, epochs=500, learning_rate=0.1, batch_size=32
)

# Evaluate simple model
simple_accuracy, simple_predictions, simple_probs, simple_loss = evaluate_tensorflow_model(
    simple_tf_model, X_dev_norm, Y_dev_categorical
)

print(f"\nSimple TensorFlow Model Results:")
print(f"Development Set Accuracy: {simple_accuracy:.4f}")
print(f"Development Set Loss: {simple_loss:.4f}")

# Visualize simple model predictions
visualize_tensorflow_predictions(simple_tf_model, X_dev_norm, Y_dev_categorical, num_samples=3)

# Advanced TensorFlow model (with improvements)
print("\n" + "="*50)
print("TENSORFLOW IMPLEMENTATION - ADVANCED MODEL")
print("="*50)

advanced_tf_model = create_advanced_tensorflow_model()
advanced_tf_model, advanced_history = train_tensorflow_model(
    advanced_tf_model, X_train_norm, Y_train_categorical, 
    X_dev_norm, Y_dev_categorical, epochs=200, learning_rate=0.01, batch_size=32
)

# Evaluate advanced model
advanced_accuracy, advanced_predictions, advanced_probs, advanced_loss = evaluate_tensorflow_model(
    advanced_tf_model, X_dev_norm, Y_dev_categorical
)

print(f"\nAdvanced TensorFlow Model Results:")
print(f"Development Set Accuracy: {advanced_accuracy:.4f}")
print(f"Development Set Loss: {advanced_loss:.4f}")

# Plot training history for advanced model
plot_training_history(advanced_history)

# Visualize advanced model predictions
visualize_tensorflow_predictions(advanced_tf_model, X_dev_norm, Y_dev_categorical, num_samples=3)

# Save models
simple_tf_model.save('mnist_tensorflow_simple_model.h5')
advanced_tf_model.save('mnist_tensorflow_advanced_model.h5')
print("\nTensorFlow models saved as 'mnist_tensorflow_simple_model.h5' and 'mnist_tensorflow_advanced_model.h5'")

print("\nTensorFlow implementation completed!")

# Performance comparison
print("\n" + "="*60)
print("TENSORFLOW MODEL COMPARISON")
print("="*60)
print(f"Simple Model  - Accuracy: {simple_accuracy:.4f}, Loss: {simple_loss:.4f}")
print(f"Advanced Model - Accuracy: {advanced_accuracy:.4f}, Loss: {advanced_loss:.4f}")
