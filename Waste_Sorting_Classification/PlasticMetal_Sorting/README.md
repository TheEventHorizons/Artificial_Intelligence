# Plastic and Metal Material Classification README

This repository contains a Python script for generating synthetic data and building a classification model to classify materials into either "Metal" or "Plastic" based on their density.

## Prerequisites

Make sure you have the following libraries installed:

- pandas
- numpy
- matplotlib
- scikit-learn

You can install these libraries using the following command:

```bash
pip install pandas numpy matplotlib scikit-learn
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/TheEventHorizons/Waste-Sorting-Classification.git
cd PlasticMetal-Sorting
```

2. Run the script:

```bash
python "Python_Script".py
```

The script preprocesses the data, trains three different models: K-Nearest Neighbors (KNN), Logistic Regression and SGDclassifier using a standard scaler pipeline, and evaluates the model's accuracy.

## Script Overview

- `"Python_Script".py`: The main Python script.
- `README.md`: This README file providing instructions and an overview of the script.
- `PlasticMetal_samples.xlsx`: This Excel file contains a comprehensive dataset comprising approximately 66,000 entries on various Metal and Plastic materials, organized based on their density.

## Data Preparation

The script reads data from an Excel file containing information about materials, such as density and type.

## Model Training

The script uses a K-Nearest Neighbors (KNN) classifier with a standard scaler pipeline. It performs a grid search for hyperparameter tuning and evaluates the model using cross-validation.

## Result

The script outputs the best hyperparameters found during the grid search, the corresponding cross-validation score, and the accuracy of the final model on the test set.

Feel free to modify the script or experiment with different classifiers to further enhance the model's performance.

For a simple explanation of how the classification works, visit my [scientific outreach website](https://theeventhorizons.com/waste-allocation-load-lifter-earth-class/) for accessible content on the topic.

Happy coding!



