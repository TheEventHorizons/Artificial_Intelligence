# Deep Learning Benchmark README

This repository contains a basic deep learning benchmark using TensorFlow and Keras. The benchmark involves image classification on a dataset of 24x24 grayscale images. The README provides an overview of the project, its structure, and steps to reproduce the benchmark.

## Project Overview

- **Objective:** Image classification using a Convolutional Neural Network (CNN) on a dataset of enhanced 24x24 grayscale images.

- **Dataset:** The dataset is stored in an HDF5 file (`set-24x24-L.h5`) and contains training, testing, and metadata sets.

- **Model Architecture:** The CNN model architecture is defined in the `model_v1` function. It consists of convolutional layers, max-pooling layers, dropout layers, and fully connected layers.

- **Training:** The model is trained using the Adam optimizer and sparse categorical cross-entropy loss. Training history is stored in the `history` variable.

## Project Structure

- `archive/data`: Directory containing the enhanced dataset in HDF5 format.
- `model.h5`: Saved trained model.
- `CNN_Benchmark_preprocessing.py`: Preprocessing script for image enhancement and dataset reading.
- `README.md`: Project documentation.
- Other necessary Python scripts and dependencies.

## Instructions

1. **Enhance Images and Read Dataset:**
   - Run the `CNN_Benchmark_preprocessing.py` script to enhance images and read the dataset.

2. **Load and Train Model:**
   - Execute the main code to load the enhanced dataset and train the CNN model.

```python
python main_script.py
```

3. **Evaluate Model:**
   - The script will print the test loss and accuracy after training completion.

4. **Saved Model:**
   - The trained model will be saved as `model.h5`.

## Requirements

- TensorFlow
- Keras
- Pillow (PIL)
- NumPy
- Matplotlib

## Notes

- Ensure that the enhanced dataset (`set-24x24-L.h5`) is available in the `archive/data` directory.
- Adjust hyperparameters such as batch size and epochs in the main script according to your requirements.

Feel free to modify and experiment with the code to suit your specific use case or dataset. Happy deep learning!