# Convolutional Neural Network (CNN) for MNIST Classification README

This repository contains a script for building and training a Convolutional Neural Network (CNN) for classifying digits from the MNIST dataset. The README provides an overview of the project, its structure, and the steps to train the model and visualize the results.

## Project Overview

- **Objective:** Develop a CNN to classify digits from the MNIST dataset.

- **Datasets:** The MNIST dataset is used, consisting of grayscale images of handwritten digits (28x28 pixels).

- **Model Architecture:**
  - Input Layer: 28x28x1 (one channel for grayscale images)
  - Convolutional Layer 1: 8 filters with 3x3 kernel, ReLU activation, MaxPooling, and Dropout
  - Convolutional Layer 2: 16 filters with 3x3 kernel, ReLU activation, MaxPooling, and Dropout
  - Flatten Layer
  - Dense (Fully Connected) Layer: 150 units, ReLU activation, and Dropout
  - Output Layer: 10 units (for digits 0-9), Softmax activation

## Project Structure

- `main_script.py`: Main script for building, training, and evaluating the CNN.
- `README.md`: Project documentation.
- Other necessary Python scripts and dependencies.

## Instructions

1. **Run the Script:**
   - Execute the `main_script.py` to build, train, and evaluate the CNN. Adjust the script parameters, such as `hidden1`, `batch_size`, and `epochs`, as needed.

```bash
python main_script.py
```

2. **Model Summary:**
   - View the summary of the CNN architecture, including the number of parameters in each layer.

3. **Training and Evaluation:**
   - The script loads the MNIST dataset, normalizes pixel values, and trains the CNN. Training and validation metrics are displayed, and the final evaluation results are printed.

4. **Visualize Training History:**
   - Visualize the training and validation accuracy and loss over epochs using Matplotlib.

5. **Confusion Matrix:**
   - A confusion matrix is generated to analyze the performance of the model on the test set. The matrix is visualized using Seaborn.

## Requirements

- NumPy
- Matplotlib
- TensorFlow
- Keras
- scikit-learn
- Seaborn

Feel free to modify and experiment with the code to suit your specific use case or dataset. The script serves as a foundation for training and evaluating CNNs for digit classification tasks. Happy coding!