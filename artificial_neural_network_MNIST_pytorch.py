import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt
import time

# Set device for computation (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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
X_train_norm = X_train / 255.0
X_dev_norm = X_dev / 255.0

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_norm).to(device)
Y_train_tensor = torch.LongTensor(Y_train).to(device)
X_dev_tensor = torch.FloatTensor(X_dev_norm).to(device)
Y_dev_tensor = torch.LongTensor(Y_dev).to(device)

print(f"Training data shape: {X_train_tensor.shape}")
print(f"Training labels shape: {Y_train_tensor.shape}")


class MNISTNet(nn.Module):
    """
    PyTorch Neural Network for MNIST Classification
    Architecture: Input(784) -> Hidden(10) -> Output(10)
    """
    def __init__(self, input_size=784, hidden_size=10, output_size=10):
        super(MNISTNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x  # No softmax here - CrossEntropyLoss includes it


def train_pytorch_model(model, X_train, Y_train, X_dev, Y_dev, epochs=500, learning_rate=0.1):
    """
    Train the PyTorch model using modern optimization techniques
    """
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Create data loaders for batch processing
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    print("\nStarting PyTorch training...")
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # Training loop
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Print progress every 10 epochs
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                dev_outputs = model(X_dev)
                _, predicted = torch.max(dev_outputs.data, 1)
                accuracy = (predicted == Y_dev).float().mean()
                print(f"Epoch {epoch:3d} | Loss: {total_loss/len(train_loader):.4f} | Dev Accuracy: {accuracy:.4f}")
    
    training_time = time.time() - start_time
    print(f"\nPyTorch training completed in {training_time:.2f} seconds")
    
    return model


def evaluate_pytorch_model(model, X_test, Y_test):
    """
    Evaluate the trained PyTorch model
    """
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == Y_test).float().mean()
        
        # Get probabilities for analysis
        probabilities = F.softmax(outputs, dim=1)
        
    return accuracy, predicted, probabilities


def visualize_pytorch_predictions(model, X_test, Y_test, num_samples=5):
    """
    Visualize sample predictions from PyTorch model
    """
    model.eval()
    with torch.no_grad():
        outputs = model(X_test[:num_samples])
        _, predictions = torch.max(outputs.data, 1)
        probabilities = F.softmax(outputs, dim=1)
    
    # Convert back to numpy for visualization
    X_np = X_test[:num_samples].cpu().numpy()
    predictions_np = predictions.cpu().numpy()
    Y_np = Y_test[:num_samples].cpu().numpy()
    probs_np = probabilities.cpu().numpy()
    
    print("\nPyTorch Sample Predictions:")
    for i in range(num_samples):
        print(f"Sample {i+1}: Predicted={predictions_np[i]}, Actual={Y_np[i]}, "
              f"Confidence={probs_np[i][predictions_np[i]]:.3f}")
        
        # Display image
        image = X_np[i].reshape(28, 28) * 255
        plt.figure(figsize=(4, 4))
        plt.gray()
        plt.imshow(image, interpolation='nearest')
        plt.title(f"PyTorch: Pred={predictions_np[i]}, True={Y_np[i]}")
        plt.axis('off')
        plt.show()


# Initialize and train PyTorch model
print("\n" + "="*50)
print("PYTORCH IMPLEMENTATION")
print("="*50)

pytorch_model = MNISTNet().to(device)
print(f"Model architecture:\n{pytorch_model}")

# Count parameters
total_params = sum(p.numel() for p in pytorch_model.parameters())
print(f"Total parameters: {total_params}")

# Train the model
trained_pytorch_model = train_pytorch_model(
    pytorch_model, X_train_tensor, Y_train_tensor, 
    X_dev_tensor, Y_dev_tensor, epochs=500, learning_rate=0.1
)

# Evaluate the model
pytorch_accuracy, pytorch_predictions, pytorch_probs = evaluate_pytorch_model(
    trained_pytorch_model, X_dev_tensor, Y_dev_tensor
)

print(f"\nFinal PyTorch Development Set Accuracy: {pytorch_accuracy:.4f}")

# Visualize predictions
visualize_pytorch_predictions(trained_pytorch_model, X_dev_tensor, Y_dev_tensor, num_samples=3)

# Save the model
torch.save(trained_pytorch_model.state_dict(), 'mnist_pytorch_model.pth')
print("PyTorch model saved as 'mnist_pytorch_model.pth'")

print("\nPyTorch implementation completed!")
