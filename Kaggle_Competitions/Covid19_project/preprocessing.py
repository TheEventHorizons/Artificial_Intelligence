import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree   import DecisionTreeClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import learning_curve

from sklearn.ensemble import RandomForestClassifier
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


# Definition of the missing rate
missing_rate = df.isna().sum()/df.shape[0]

# Creation of the Blood and viral columns
blood_columns = list(df.columns[(missing_rate < 0.9) & (missing_rate > 0.88)])
viral_columns = list(df.columns[(missing_rate < 0.88) & (missing_rate > 0.75)])

# Creation of important columns
key_columns = ['Patient age quantile', 'SARS-Cov-2 exam result']

# Filter, remove 'Influenza A/B, rapid test' and display the new the dataframe 
df = df[key_columns+blood_columns+viral_columns]
df = df.drop(['Influenza B, rapid test', 'Influenza A, rapid test'], axis=1)
print(df.head())







##############################################################################################
#                                           TRAIN TEST 
##############################################################################################



# Create the train_set and the test_set
train_set, test_set = train_test_split(df, test_size=0.2, random_state = 0)

# Display the quantity of negative and positive values in the 'SARS-Cov-2 exam result' column of the train_set
print(train_set['SARS-Cov-2 exam result'].value_counts())

# Display the quantity of negative and positive values in the 'SARS-Cov-2 exam result' column of the test_set
print(test_set['SARS-Cov-2 exam result'].value_counts())




##############################################################################################
#                                           ENCODING
##############################################################################################


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

# Display the new data frame
print(encoder(df).head())

# Are there object type variables ? 
print(encoder(df).dtypes.value_counts())
'''
# Create an imputation function (the most basic one)
def imputation(df):
    return df.dropna(axis=0)

# Create a preprocessing function
def preprocessing(df):
    df = encoder(df)
    df = imputation(df)

    X = df.drop('SARS-Cov-2 exam result', axis=1)
    y = df['SARS-Cov-2 exam result']

    print(y.value_counts())
    return X, y


# Create the X_train, y_train and the X_test, y_test
X_train, y_train = preprocessing(train_set)
X_test, y_test = preprocessing(test_set)
'''




##############################################################################################
#                       INITIAL MODELING AND EVALUATION PROCEDURE
##############################################################################################

'''
# A simple way of testing is with decision trees

# Create a decision tree model
model = DecisionTreeClassifier(random_state=0)

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


# Overfitting !!!
#evaluation(model)
'''



##############################################################################################
#                       SECOND MODELING AND EVALUATION PROCEDURE
##############################################################################################


'''
# Create another imputation function, for example fillna with an extreme value
def imputation(df):
    return df.fillna(-999)

# Create a preprocessing function
def preprocessing(df):
    df = encoder(df)
    df = imputation(df)

    X = df.drop('SARS-Cov-2 exam result', axis=1)
    y = df['SARS-Cov-2 exam result']

    print(y.value_counts())
    return X, y


# Create the X_train, y_train and the X_test, y_test
X_train, y_train = preprocessing(train_set)
X_test, y_test = preprocessing(test_set)

# A simple way of testing is with decision trees

# Create a decision tree model
model = DecisionTreeClassifier(random_state=0)

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

# Evaluate the model: It's worse, fillna is not a good imputation
evaluation(model)
'''



##############################################################################################
#                       THIRD MODELING AND EVALUATION PROCEDURE
##############################################################################################


# Go back to our first model
'''
# Create an imputation function (the most basic one)
def imputation(df):
    return df.dropna(axis=0)

# Create a preprocessing function
def preprocessing(df):
    df = encoder(df)
    df = imputation(df)

    X = df.drop('SARS-Cov-2 exam result', axis=1)
    y = df['SARS-Cov-2 exam result']

    print(y.value_counts())
    return X, y


# Create the X_train, y_train and the X_test, y_test
X_train, y_train = preprocessing(train_set)
X_test, y_test = preprocessing(test_set)

# A simple way of testing is with decision trees

# Create a decision tree model
model = DecisionTreeClassifier(random_state=0)



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


# Overfitting !!!
evaluation(model)


# Start by selecting important features
pd.DataFrame(model.feature_importances_, index=X_train.columns).plot.bar(figsize=(12,8))
plt.show()'''

# Two option - eliminate all viral columns (which seem to be not significant) or - take a threshold to eliminate all variables below for example 0.01  

##################### The first one #####################
'''
# Filter and display the new the dataframe 
df = df[key_columns+blood_columns]
print(df.head())

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


# Create an imputation function (the most basic one)
def imputation(df):
    return df.dropna(axis=0)

# Create a preprocessing function
def preprocessing(df):
    df = encoder(df)
    df = imputation(df)

    X = df.drop('SARS-Cov-2 exam result', axis=1)
    y = df['SARS-Cov-2 exam result']

    print(y.value_counts())
    return X, y


# Create the X_train, y_train and the X_test, y_test
X_train, y_train = preprocessing(train_set)
X_test, y_test = preprocessing(test_set)

# A simple way of testing is with decision trees

# Create a decision tree model
model = DecisionTreeClassifier(random_state=0)



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


# Overfitting again !!!
evaluation(model)
'''


##############################################################################################
#                       FOURTH MODELING AND EVALUATION PROCEDURE
##############################################################################################

'''
# On essaye d'éliminer l'overfitting. On va utiliser un randomforestclassifier

# When we created a variable 'is sick' we saw significant differences in blood tests (Lymphocytes) something that we do not have with covid
# it's possible to reach best results by creating a variable 'is sick'

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
model = RandomForestClassifier(random_state=0)

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


# Overfitting again !!!
evaluation(model)

# Start by selecting important features for this model
pd.DataFrame(model.feature_importances_, index=X_train.columns).plot.bar(figsize=(12,8))
plt.show()


## Not better. We will try to take the most useful variables with a selectKBest with an anova test for example




##############################################################################################
#                       FIFTH MODELING AND EVALUATION PROCEDURE
##############################################################################################


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
model = make_pipeline(SelectKBest(f_classif, k = 5),
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
evaluation(model)
'''



##############################################################################################
#                       SIXTH MODELING AND EVALUATION PROCEDURE
##############################################################################################


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
evaluation(model)




