#!/usr/bin/env python3
"""
Simple runner script for the MNIST Neural Network
This script provides a clean way to run the neural network with error handling
"""

import os
import sys

def check_dependencies():
    """Check if all required packages are installed"""
    try:
        import numpy
        import pandas 
        import matplotlib
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def check_data_files():
    """Check if MNIST data files exist"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(current_dir, "mnist_train.csv")
    test_file = os.path.join(current_dir, "mnist_test.csv")
    
    if not os.path.exists(train_file):
        print(f"❌ Missing file: {train_file}")
        print("Please download mnist_train.csv from Kaggle MNIST dataset")
        return False
    
    if not os.path.exists(test_file):
        print(f"❌ Missing file: {test_file}")
        print("Please download mnist_test.csv from Kaggle MNIST dataset")
        return False
    
    print("✅ All data files found")
    return True

def main():
    """Main runner function"""
    print("🚀 MNIST Neural Network Runner")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check data files
    if not check_data_files():
        print("\n📖 Setup Instructions:")
        print("1. Visit: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv")
        print("2. Download mnist_train.csv and mnist_test.csv")
        print("3. Place both files in this directory")
        sys.exit(1)
    
    # Run the neural network
    print("\n🧠 Starting Neural Network Training...")
    print("This may take 2-5 minutes depending on your hardware")
    print("-" * 40)
    
    try:
        # Import and run the main script
        import artificial_neural_network_MNIST
        print("\n🎉 Training completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        print("Please check the error message above and refer to setup_guide.md for troubleshooting")
        sys.exit(1)

if __name__ == "__main__":
    main()
