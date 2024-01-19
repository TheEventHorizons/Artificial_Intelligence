# Facial Expression Recognition with Data Augmentation

This repository extends the facial expression recognition preprocessing scripts to include training a convolutional neural network (CNN) model for emotion classification. The code is written in Python, utilizing TensorFlow and Keras for deep learning tasks.

## Prerequisites

Make sure to install the required libraries before running the scripts:

```bash
pip install numpy matplotlib pandas h5py scikit-image seaborn tensorflow
```

## Reduce Training Set and Create Train and Validation Folders

The `reduce_training_set.py` script reduces the original dataset to 10%, while `create_train_validation_folders.py` organizes images into training and validation sets.

### Usage

1. Set the `data_dir` variable in both scripts to the path of your original dataset.
2. Run the scripts:

```bash
python reduce_training_set.py
python create_train_validation_folders.py
```

## Train the Facial Expression Recognition Model

The `train_facial_expression_model.py` script trains a CNN model using data augmentation for improved performance.

### Usage

1. Set the appropriate variables such as `data_dir_train`, `data_dir_val`, and others.
2. Run the script:

```bash
python train_facial_expression_model.py
```

### Additional Features

- The script includes callbacks for TensorBoard, ModelCheckpoint to save the best model, and ModelCheckpoint to save the model at each epoch.
- You can monitor the training progress using TensorBoard. Use the provided TensorBoard command after running the script.
- The training history and model summary are printed for analysis.

Feel free to adapt file paths, parameters, or add more customization based on your needs. If you encounter any issues or have questions, don't hesitate to ask!