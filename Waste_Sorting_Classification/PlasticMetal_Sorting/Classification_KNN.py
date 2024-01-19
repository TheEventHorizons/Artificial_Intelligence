import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import os
print("Current Working Directory:", os.getcwd())

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import  StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve, learning_curve, cross_val_score
from sklearn.preprocessing import LabelEncoder



# Replace 'path/to/your/excel/file.xlsx' with the actual path to your Excel file
excel_file_path = 'PlasticMetal_samples.xlsx'

# Read the Excel file
result_df = pd.read_excel(excel_file_path)

# Display the Pandas DataFrame
print(result_df)


X = result_df['Density']
y = result_df['Type of Material']




# Data preparation


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=42)


# Convert labels to the appropriate format for binary classification
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Reshape X_train and X_test
X_train_reshaped = X_train.values.reshape(-1, 1)  # Reshape into a single column
X_test_reshaped = X_test.values.reshape(-1, 1)    # Reshape into a single column


# Create the model
model = make_pipeline(StandardScaler(), KNeighborsClassifier())

print(model)

# Create parameter dictionary

params = {
    'kneighborsclassifier__n_neighbors':[2,3,4,5,6,7,8,9,10, 15, 20],
    'kneighborsclassifier__weights': ['uniform', 'distance'],
    'kneighborsclassifier__p': [1, 2]
}



# Create a search grid

grid = GridSearchCV(model, param_grid= params, cv = 5)


# Train the model
grid.fit(X_train_reshaped, y_train_encoded)

print(grid.best_params_)
print(grid.best_score_)

modelB = grid.best_estimator_

# Model evaluation
accuracy = modelB.score(X_test_reshaped, y_test_encoded)
print(f"Model accuracy : {accuracy}")

# Predictions
predictions = modelB.predict(X_test_reshaped)

print(predictions)

# Validation Curves

k_range = np.arange(1,20)

train_score, val_score = validation_curve(model, X_train_reshaped, y_train_encoded, param_name='kneighborsclassifier__n_neighbors',param_range=k_range, cv =5)


plt.plot(k_range,val_score.mean(axis=1), label='Validation')
plt.plot(k_range,train_score.mean(axis=1), label='Train')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Score')
plt.grid(ls='--')
plt.legend()
plt.title('Validation and Train Curves for k-NN')
plt.show()

# Learning curves

N, learn_train_score, learn_val_score = learning_curve(modelB, X_train_reshaped, y_train_encoded, train_sizes=np.linspace(0.01, 1.0, 50), cv = 5)

print(N)
plt.plot(N,learn_val_score.mean(axis=1), label='Validation')
plt.plot(N,learn_train_score.mean(axis=1), label='Train')
plt.xlabel('Train Sizes')
plt.ylabel('Score')
plt.grid(ls='--')
plt.legend()
plt.title('Validation and Train Learning Curves for k-NN')
plt.show()
