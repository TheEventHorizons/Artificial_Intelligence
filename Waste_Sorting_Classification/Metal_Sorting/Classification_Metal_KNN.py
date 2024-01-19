import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
print("Current Working Directory:", os.getcwd())


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve, learning_curve
from sklearn.preprocessing import LabelEncoder



# Replace 'path/to/your/excel/file.xlsx' with the actual path to your Excel file
excel_file_path = 'metals_samples.xlsx'

# Read the Excel file
result_metals_df = pd.read_excel(excel_file_path)

# Display the Pandas DataFrame
print(result_metals_df)


# Remove missing values
result_metals_df = result_metals_df.dropna(axis=0)


X = result_metals_df[['Density', 'Electrical Conductivity']]
y = result_metals_df['Metal Type']


# Data preparation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert labels to the appropriate format for binary classification
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Create the model
model = make_pipeline(MinMaxScaler(), KNeighborsClassifier())

print(model)

# Create parameter dictionary
params = {
    'kneighborsclassifier__n_neighbors': [2,3,4,5,6,7,8,9,10, 15, 20, 25, 30, 40,  100],
    'kneighborsclassifier__weights': ['uniform', 'distance'],
    'kneighborsclassifier__p': [1, 2]
}

# Create a search grid
grid = GridSearchCV(model, param_grid=params, cv=5)

# Train the model
grid.fit(X_train, y_train_encoded)

print(grid.best_params_)
print(grid.best_score_)

modelB = grid.best_estimator_

# Model evaluation
accuracy = modelB.score(X_test, y_test_encoded)
print(f"Model Accuracy: {accuracy}")

# Predictions
predictions = modelB.predict(X_test)


#for i in range(0, 32):
    #print(label_encoder.inverse_transform(np.array([i])))



# Sample
X_sample = pd.DataFrame({'Density': [7.5], 'Electrical Conductivity': [5.8], 'Metal Type' : [np.NaN]})  
X_sample = X_sample[['Density','Electrical Conductivity']]


# Predict using the scaled sample
prediction = modelB.predict(X_sample)
prediction_proba = modelB.predict_proba(X_sample)

print("Predicted Metal Type:", label_encoder.inverse_transform(prediction))
print("Prediction Probabilities:", prediction_proba)

# Probability Prediction
prediction_proba = modelB.predict_proba(X_sample)


# Validation Curves

k_range = np.arange(1,100)

train_score, val_score = validation_curve(model, X_train, y_train_encoded, param_name='kneighborsclassifier__n_neighbors',param_range=k_range, cv =5)

plt.plot(k_range,val_score.mean(axis=1), label='Validation')
plt.plot(k_range,train_score.mean(axis=1), label='Train')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Score')
plt.grid(ls='--')
plt.legend()
plt.title('Validation and Train Curves for k-NN')
plt.show()

# Learning curves

N, learn_train_score, learn_val_score = learning_curve(modelB, X_train, y_train_encoded, train_sizes=np.linspace(0.01, 1.0, 50), cv = 5)

print(N)
plt.plot(N,learn_val_score.mean(axis=1), label='Validation')
plt.axhline(y=0.957, c='red', ls='--', label = 'Threshold')
plt.plot(N,learn_train_score.mean(axis=1), label='Train')
plt.xlabel('Train Sizes')
plt.ylabel('Score')
plt.grid(ls='--')
plt.legend()
plt.title('Validation and Train Learning Curves for k-NN')
plt.show()


# Displaying probabilities with class names
for i, metal_class in enumerate(label_encoder.classes_):
    print(f"Probability for class '{metal_class}': {prediction_proba[0, i]}")





plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_train['Density'], X_train['Electrical Conductivity'], c=y_train_encoded, cmap='Dark2_r', alpha=0.5)
plt.scatter(X_sample['Density'], X_sample['Electrical Conductivity'], c='red', s=50, marker='X', alpha=0.8)
plt.axvline(x=tuple(X_sample['Density']), c='red', ls='--', alpha=0.8)
plt.axhline(y=tuple(X_sample['Electrical Conductivity']), c='red', ls='--', alpha=0.8)
plt.xlabel('Density')
plt.ylabel('Electrical Conductivity (S/m)')
plt.grid(ls='--')
# Configure the colorbar correctly
cbar = plt.colorbar(scatter, ticks=range(len(label_encoder.classes_)))
cbar.set_label('Metal Alloys')
cbar.set_ticklabels(label_encoder.classes_)
plt.title('Unknown Metallic Sample Based on Density and Electrical Conductivity')
plt.legend()
plt.show()
