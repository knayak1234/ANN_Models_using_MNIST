"""
Configuration file for MNIST Neural Network
Modify these parameters to experiment with different settings
"""

# Training Parameters
LEARNING_RATE = 0.1      # Learning rate for gradient descent
ITERATIONS = 500         # Number of training iterations
PRINT_EVERY = 10        # Print accuracy every N iterations

# Data Parameters
DEV_SIZE = 1000         # Number of samples for development set
TRAIN_RATIO = 0.9       # Ratio of data to use for training

# Network Architecture
HIDDEN_LAYER_SIZE = 10  # Number of neurons in hidden layer
INPUT_SIZE = 784        # Input size (28x28 = 784 pixels)
OUTPUT_SIZE = 10        # Output size (10 digits: 0-9)

# Data Paths
TRAIN_DATA_PATH = "mnist_train.csv"
TEST_DATA_PATH = "mnist_test.csv"

# Visualization Parameters
SHOW_PREDICTIONS = True  # Whether to show sample predictions
NUM_SAMPLES_TO_SHOW = 5  # Number of sample predictions to display

# Random Seed (for reproducible results)
RANDOM_SEED = 42        # Set to None for random initialization

# Advanced Parameters
NORMALIZE_DATA = True    # Whether to normalize pixel values to 0-1
SHUFFLE_DATA = True      # Whether to shuffle data before splitting
USE_ONE_HOT = True       # Whether to use one-hot encoding for labels

# Performance Settings
BATCH_SIZE = None        # Batch size (None = full batch)
EARLY_STOPPING = False   # Whether to stop early if accuracy is high
EARLY_STOP_THRESHOLD = 0.95  # Accuracy threshold for early stopping

# Debug Settings
DEBUG_MODE = False       # Enable debug prints
VERBOSE = True          # Print detailed information
