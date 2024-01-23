import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree   import DecisionTreeClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report, precision_recall_curve, recall_score

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import learning_curve, GridSearchCV, RandomizedSearchCV

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.pipeline import make_pipeline


data_path = '/Users/jordanmoles/Documents/GitHub/Artificial_Intelligence/Covid-19_project/dataset.xlsx'



# Display the max row and the max columns
pd.set_option('display.max_row',111)
#pd.set_option('display.max_columns',111)

# Read the data
data = pd.read_excel(data_path)

# Copy the Data
df = data.copy()

# Display few rows
print(df.head())


# Création des sous-ensembles (suite au EDA)


# Definition of the missing rate
missing_rate = df.isna().sum()/df.shape[0]

# Creation of the Blood and viral columns
blood_columns = list(df.columns[(missing_rate < 0.9) & (missing_rate > 0.88)])
viral_columns = list(df.columns[(missing_rate < 0.88) & (missing_rate > 0.75)])

# Creation of important columns
key_columns = ['Patient age quantile', 'SARS-Cov-2 exam result']


# Add polynomial variables

# Copy the Data
df = data.copy()

# Filter, remove 'Influenza A/B, rapid test' and display the new the dataframe 
df = df[key_columns+blood_columns+viral_columns]

# Create the train_set and the test_set
train_set, test_set = train_test_split(df, test_size=0.2, random_state = 0)

# Create an encoding function
def encoder(df):
    code = {'positive': 1,
            'negative': 0,
            'detected': 1,
            'not_detected':0}
    # Encode the train_set
    for col in df.select_dtypes('object').columns:
        df.loc[:,col] = df[col].map(code)

    return df


# Create a feature_engineering function
def feature_engineering(df):
    # Create a 'is sick' columns to detect if people is detected to another virus
    df['is sick'] = df[viral_columns].sum(axis=1)>=1
    # Remove all viral_columns
    df = df.drop(viral_columns, axis=1)
    return df

# Create an imputation function (the most basic one)
def imputation(df):
    return df.dropna(axis=0)

# Create a preprocessing function
def preprocessing(df):
    df = encoder(df)
    df = feature_engineering(df)
    df = imputation(df)

    X = df.drop('SARS-Cov-2 exam result', axis=1)
    y = df['SARS-Cov-2 exam result']

    print(y.value_counts())
    return X, y


# Create the X_train, y_train and the X_test, y_test
X_train, y_train = preprocessing(train_set)
X_test, y_test = preprocessing(test_set)

# Create a decision tree model
model = make_pipeline(PolynomialFeatures(2),SelectKBest(f_classif, k = 10),
                      RandomForestClassifier(random_state=0))

# Create evaluation function
def evaluation(model):
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    # Display confusion matrix and classification report
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Create learning curves
    N, train_score, val_score = learning_curve(model, X_train, y_train, cv = 4, scoring = 'f1', train_sizes=np.linspace(0.1,1,10))
    plt.figure(figsize=(12,8))
    plt.plot(N, train_score.mean(axis=1), label='train_score', c='blue')
    plt.plot(N, val_score.mean(axis=1), label='val_score', c='orange')
    plt.grid(ls='--')
    plt.legend()
    plt.show()


# Overfitting again but it's better!!!
#evaluation(model)



##############################################################################################
#                                     MODEL IMPROVEMENT
##############################################################################################

# Improve the model's performance based on the adjustments we made by changing the chosen model type

# Creating the preprocessor pipeline to add upstream
preprocessor = make_pipeline(PolynomialFeatures(2, include_bias=False), SelectKBest(f_classif, k=10))

# Creating model pipelines
RandomForest = make_pipeline(preprocessor, RandomForestClassifier(random_state=0))
AdaBoost = make_pipeline(preprocessor, AdaBoostClassifier(random_state=0))
# Adding normalization for the following models (the first two do not need normalized datasets)
SVM = make_pipeline(preprocessor, StandardScaler(), SVC(random_state=0))
KNN = make_pipeline(preprocessor, StandardScaler(), KNeighborsClassifier())

# Creating a dictionary of models
dict_of_models = {'RandomForest': RandomForest, 'AdaBoost': AdaBoost, 'SVM': SVM, 'KNN': KNN}

# Evaluating each model
'''
for name, model in dict_of_models.items():
    print(name)
    evaluation(model)
'''
# SVM and KNN are good choices because they are not overfitting. SVM may be more suitable.






##############################################################################################
#                                     MODEL OPTIMIZATION
##############################################################################################
    
'''
# Create hyperparameters dictionary
hyper_params = {'svc__gamma': [1e-3, 1e-4],
                'svc__C': [1, 10, 100, 1000]}

# Create a grid
grid = GridSearchCV(SVM, hyper_params, scoring='recall', cv=4)

# Train the grid
grid.fit(X_train, y_train)

# Display the best parameters
print(grid.best_params_)

# Calculate a prediction vector and compare to other values
y_pred = grid.predict(X_test)
print(classification_report(y_test, y_pred))

# Evaluate the model
evaluation(grid.best_estimator_)

# Same score (not very good) more or less on the train set and the validation set. Good!
'''





##############################################################################################
#                                     SECOND MODEL OPTIMIZATION 
##############################################################################################
    
# Further improvement without testing 20,000 parameters with GridSearchCV. We can use RandomizedSearchCV

# Create hyperparameters dictionary
hyper_params = {'svc__gamma': [1e-3, 1e-4],
                'svc__C': [1, 10, 100, 1000], 
                'pipeline__polynomialfeatures__degree': [2, 3, 4],
                'pipeline__selectkbest__k': range(40, 60)}

# Create a grid that randomly searches for the best hyperparameters
grid = RandomizedSearchCV(SVM, hyper_params, scoring='recall', cv=4, n_iter=40)

# Train the grid
grid.fit(X_train, y_train)

# Display the best parameters
print(grid.best_params_)

# Calculate a prediction vector and compare to other values
y_pred = grid.predict(X_test)
print(classification_report(y_test, y_pred))

# Evaluate the model
evaluation(grid.best_estimator_)



##############################################################################################
#                                     FINALISATION DE NOTRE MODELE 
##############################################################################################


# Observing precision/recall curves and defining a prediction threshold for the model (for binary classification)
precision, recall, threshold = precision_recall_curve(y_test, grid.best_estimator_.decision_function(X_test))

# Display the figures
plt.figure(figsize=(12, 8))
plt.plot(threshold, precision[:-1], label='precision')
plt.plot(threshold, recall[:-1], label='recall')
plt.legend()
plt.grid(ls='--')
plt.show()

# Define prediction function
def final_model(model, X, threshold=0):
    return model.decision_function(X) > threshold

# Using the defined prediction function with a specific threshold
y_pred = final_model(grid.best_estimator_, X_test, threshold=-1)

# Print F1 score and recall score for the final model
print(f1_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
