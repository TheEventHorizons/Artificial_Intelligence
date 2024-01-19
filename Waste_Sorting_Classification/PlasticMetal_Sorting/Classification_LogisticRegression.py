import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
print("Current Working Directory:", os.getcwd())

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import OrdinalEncoder




# Replace 'path/to/your/excel/file.xlsx' with the actual path to your Excel file
excel_file_path = 'PlasticMetal_samples.xlsx'

# Read the Excel file
result_df = pd.read_excel(excel_file_path)

# Display the Pandas DataFrame
print(result_df)


X = result_df['Density']
y = result_df['Type of Material']

# Data preparation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert labels to the appropriate format for binary classification
ordinal_encoder = OrdinalEncoder(categories=[['Plastic', 'Metal']])
y_train_encoded = ordinal_encoder.fit_transform(y_train.values.reshape(-1, 1))
y_test_encoded = ordinal_encoder.transform(y_test.values.reshape(-1, 1))

# Reshape y_train and y_test
y_train_reshaped = y_train_encoded.ravel()
y_test_reshaped = y_test_encoded.ravel()

# Reshape X_train and X_test
X_train_reshaped = X_train.values.reshape(-1, 1)
X_test_reshaped = X_test.values.reshape(-1, 1)

# Create the model
model = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))

print(model)

# Create parameter dictionary
params = {
    'logisticregression__penalty': ['l2', None],
    'logisticregression__solver': ['lbfgs'],
    'logisticregression__max_iter': [100, 200, 300, 500, 1000]
}

# Create a search grid
grid = GridSearchCV(model, param_grid=params, cv=5)

# Train the model
grid.fit(X_train_reshaped, y_train_reshaped)

print(grid.best_params_)
print(grid.best_score_)

modelB = grid.best_estimator_

# Model evaluation
accuracy = modelB.score(X_test_reshaped, y_test_reshaped)
print(f"Model accuracy: {accuracy}")

# Predictions
predictions = modelB.predict(X_test_reshaped)

print(predictions)
print(ordinal_encoder.inverse_transform(np.array([[1], [0]])))

coefficients = modelB.named_steps['logisticregression'].coef_
intercept = modelB.named_steps['logisticregression'].intercept_

print("Coefficients:", coefficients)
print("Intercept:", intercept)

# Plot the sigmoid function with normalized data
density_values = np.linspace(0, X_train_reshaped.max(), 300).reshape(-1, 1)
probabilities = modelB.predict_proba(density_values)[:, 1]

# Find the density corresponding to a probability of 0.5
decision_boundary = density_values[np.argmax(probabilities >= 0.5)]

# Plot the sigmoid function
plt.figure(figsize=(10, 6))
plt.scatter(X_train_reshaped, y_train_reshaped, c=y_train_reshaped, cmap='Dark2_r', alpha=0.5)
plt.plot(density_values, probabilities, color='blue', linewidth=2, label='Sigmoid Function')
# Add the vertical line corresponding to a probability of 0.5
plt.axvline(x=decision_boundary, color='red', linestyle='--', label='Decision Boundary')
plt.title('Final Representation of the Sigmoid Function')
plt.xlabel('Density')
plt.ylabel('Probability of Metal')
plt.grid(ls='--')
plt.legend()
plt.show()


# Learning curves

N, learn_train_score, learn_val_score = learning_curve(modelB, X_train_reshaped, y_train_reshaped, train_sizes=np.linspace(0.01, 1.0, 50), cv = 5)

print(N)
plt.plot(N,learn_val_score.mean(axis=1), label='Validation')
plt.plot(N,learn_train_score.mean(axis=1), label='Train')
plt.xlabel('Train Sizes')
plt.ylabel('Score')
plt.grid(ls='--')
plt.legend()
plt.title('Validation and Train Learning Curves for Logistic Regression')
plt.show()

# Suppose you have a new material with a certain density
new_material_density = 4  # Replace this value with the actual density of your new material

new_material_density_reshaped = np.array(new_material_density).reshape(-1, 1)

print(modelB.predict(new_material_density_reshaped))

# Make the prediction with the model
probability_of_metal = modelB.predict_proba(new_material_density_reshaped)[:, 1]

# Display the result
print("Probability of being metal:", probability_of_metal)
