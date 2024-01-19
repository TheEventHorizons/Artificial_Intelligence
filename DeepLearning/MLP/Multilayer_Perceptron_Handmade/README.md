# DeepLearning
 
# README.md

## Description
This repository contains a simple handmade neural network implementation using NumPy and Keras to classify handwritten digits from the MNIST dataset. The neural network is trained using a basic implementation of forward and backward propagation.

## Code Overview
- Load MNIST dataset and display one image for each digit using Matplotlib.
- Reshape and normalize input data.
- One-hot encode labels and select a subset of data.
- Define functions for neural network initialization, forward propagation, backpropagation, weight update, prediction, and training.
- Train the neural network on a subset of the MNIST training data.
- Display training loss and accuracy plots.
- Evaluate the trained model on the test set and print the accuracy.

## Dependencies
- `numpy` for numerical operations.
- `matplotlib` for data visualization.
- `tqdm` for displaying progress bars.
- `scikit-learn` for metrics like log loss and accuracy.
- `keras` for loading the MNIST dataset and one-hot encoding labels.

## Instructions
1. Clone the repository: `git clone https://github.com/your-username/your-repository.git`
2. Navigate to the project directory: `cd your-repository`
3. Install dependencies: `pip install -r requirements.txt` (if available)
4. Run the script: `python your_script.py`

## Results
- Displayed images of one example for each digit from the MNIST dataset.
- Training loss and accuracy plots during the neural network training.
- Test accuracy of the trained model on the MNIST test set.

Feel free to explore and modify the code to experiment with different neural network architectures or datasets. If you encounter any issues or have suggestions, please open an issue or submit a pull request. Happy coding!