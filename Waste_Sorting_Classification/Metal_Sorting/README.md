# Metal Classification with k-NN

This repository contains a Python script for classifying metal types based on density and electrical conductivity using k-Nearest Neighbors (k-NN) classification. The script utilizes the scikit-learn library for building and evaluating the model.

## Getting Started

### Prerequisites

- Python (3.6 or higher)
- Jupyter Notebook (optional)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/TheEventHorizons/Waste-Sorting-Classification.git
```

2. Navigate to the project directory:

```bash
cd Metal-Sorting
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Replace 'metals_samples.xlsx' with the path to your Excel file containing metal samples.
2. Execute the script:

```bash
python Classification_Metal_KNN.py
```

3. View the results, including model accuracy, predictions, and visualizations.

## Script Overview

- **Data Loading:** The script loads the metal samples from the Excel file and displays the Pandas DataFrame.

- **Data Preprocessing:** Missing values are removed, and features (`Density` and `Electrical Conductivity`) and labels (`Metal Type`) are prepared.

- **Model Training:** A k-NN classifier is trained using the training set, and hyperparameter tuning is performed using GridSearchCV.

- **Model Evaluation:** The accuracy of the trained model is evaluated on the test set, and predictions are displayed.

- **Validation Curves:** Validation curves are plotted to help in choosing the optimal number of neighbors (`k`) for the k-NN classifier.

- **Learning Curves:** Learning curves are plotted to visualize how the model's performance changes with different training set sizes.

- **Probability Prediction:** The script demonstrates how to obtain probability predictions for each class.

- **Visualization:** A scatter plot is generated to visualize the metal samples in the feature space, and an unknown metallic sample is highlighted for classification.

## Result

The script provides insights into the classification model's performance, helps in selecting optimal hyperparameters, and visualizes the results for better understanding.

Feel free to modify the script according to your specific dataset and requirements.

For a simple explanation of how k-NN classification works, visit my [scientific outreach website](https://www.theeventhorizons.com) for accessible content on the topic.

Happy Coding !