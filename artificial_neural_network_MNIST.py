import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Load the data
data = pd.read_csv("/Users/knayak/AI_Startup/Learn_Software/DeepLearning/first_ANN_model/mnist_test.csv")
print(f"Original data shape: {data.shape}")

# Convert to numpy array
data = np.array(data)
m, n = data.shape  # dimensions of the data
print(f"Data dimensions: {m} samples, {n} features")

# Shuffle the data BEFORE splitting
np.random.shuffle(data)

# Split into dev and train sets
data_dev = data[0:1000].T  # First 1000 samples for development
data_train = data[1000:m].T  # Remaining samples for training

# Extract features and labels for dev set
Y_dev = data_dev[0]  # First row is labels
X_dev = data_dev[1:n]  # Remaining rows are pixel features

# Extract features and labels for train set  
Y_train = data_train[0]  # First row is labels
X_train = data_train[1:n]  # Remaining rows are pixel features

print(f"Dev set - X shape: {X_dev.shape}, Y shape: {Y_dev.shape}")
print(f"Train set - X shape: {X_train.shape}, Y shape: {Y_train.shape}")
print(f"Sample labels from training set: {Y_train[:10]}")

def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)  

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1 @ X + b1
    A1 = ReLU(Z1)
    Z2 = W2 @ A1 + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def deriv_ReLU(Z):
    return Z > 0

def back_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    m = Y.shape[1]
    dZ2 = A2 - Y
    dW2 = 1 / m * dZ2 @ A1.T
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T @ dZ2 * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1 @ X.T
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2   

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, axis=0)

def get_accuracy(predictions, Y):
    # Convert one-hot Y back to labels for comparison
    Y_labels = np.argmax(Y, axis=0)
    return np.sum(predictions == Y_labels) / Y_labels.size    


def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if(i % 10 == 0):
            print(f"Iteration: {i}")
            print(f"Accuracy: {get_accuracy(get_predictions(A2), Y)}")
    return W1, b1, W2, b2

# Normalize the input data
X_train_norm = X_train / 255.0
X_dev_norm = X_dev / 255.0

# Convert labels to one-hot encoding
Y_train_one_hot = one_hot(Y_train.astype(int))
Y_dev_one_hot = one_hot(Y_dev.astype(int))

print(f"X_train_norm shape: {X_train_norm.shape}")
print(f"Y_train_one_hot shape: {Y_train_one_hot.shape}")

# Train the neural network
W1, b1, W2, b2 = gradient_descent(X_train_norm, Y_train_one_hot, 0.1, 500)
print(f"W1 shape: {W1.shape}")
print(f"b1 shape: {b1.shape}")
print(f"W2 shape: {W2.shape}")
print(f"b2 shape: {b2.shape}")
###############################################



# Test on development set
def test_prediction(index, W1, b1, W2, b2, X, Y):
    current_image = X[:, index, None]
    prediction = make_predictions(X[:, index, None], W1, b1, W2, b2)
    label = Y[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

# Test the model on development set
print("\nTesting on development set...")
_, _, _, A2_dev = forward_prop(W1, b1, W2, b2, X_dev_norm)
dev_predictions = get_predictions(A2_dev)
dev_accuracy = get_accuracy(dev_predictions, Y_dev_one_hot)
print(f"Development set accuracy: {dev_accuracy:.4f}")

# Show a few test predictions
print("\nSample predictions:")
for i in range(5):
    test_prediction(i, W1, b1, W2, b2, X_dev_norm, Y_dev)

#print(train.head())
#print(test.head())

#print(train.shape)
#print(test.shape)

#print(train.columns)
