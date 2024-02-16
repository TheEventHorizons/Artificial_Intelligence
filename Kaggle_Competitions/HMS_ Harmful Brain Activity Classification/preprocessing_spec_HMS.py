import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import catboost as cat
from catboost import CatBoostClassifier, Pool

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.tree   import DecisionTreeClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import learning_curve

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline



data_train_path = '/Users/jordanmoles/Documents/Programmes_Informatiques/Python/Projects/Kaggle_Competitions/hms-harmful-brain-activity-classification/train.csv'



# Display the max row and the max columns
pd.set_option('display.max_row',111)
#pd.set_option('display.max_columns',111)

# Read the data
data = pd.read_csv(data_train_path)

# Copy the Data
df = data.copy()
'''
# Observe few lines 
print(df.head())

# Shape of the data
print('The shape of df is:', df.shape)

# Create columns
Column_name = list(df.columns)
'''



############################################################################################################################################################################################
#                                                                                          TRAIN SET
############################################################################################################################################################################################

# Select unique spectrograms
df_unique_spectrograms = df.drop_duplicates(subset='spectrogram_id').reset_index(drop=True)
#print(df_unique_spectrograms.head())
#print(df_unique_spectrograms.shape)



train_spec_path = '/Users/jordanmoles/Documents/Programmes_Informatiques/Python/Projects/Kaggle_Competitions/hms-harmful-brain-activity-classification/train_spectrograms/'


GET_ROW=0
row = df_unique_spectrograms.iloc[GET_ROW]
df_spectrogram = pd.read_parquet(f'{train_spec_path}{row.spectrogram_id}.parquet')
spec_offset = int( row.spectrogram_label_offset_seconds )
df_spectrogram = df_spectrogram.loc[(df_spectrogram.time>=spec_offset)
                        &(df_spectrogram.time<spec_offset+600)]
# Copy the Data
df_spectrogram = df_spectrogram.copy()
df_spectrogram = df_spectrogram.drop(['time'], axis=1)


# Calculez la moyenne de chaque colonne avec un préfixe 'mean_'
mean_values = df_spectrogram.mean()
mean_values = mean_values.add_prefix('mean_')

# Calculez le minimum de chaque colonne avec un préfixe 'min_'
min_values = df_spectrogram.min()
min_values = min_values.add_prefix('min_')

# Calculez le maximum de chaque colonne avec un préfixe 'max_'
max_values = df_spectrogram.max()
max_values = max_values.add_prefix('max_')

# Calculez l'écart type de chaque colonne avec un préfixe 'std_'
std_values = df_spectrogram.std()
std_values = std_values.add_prefix('std_')

# Concaténez les résultats pour obtenir une ligne
summary_row = pd.concat([min_values, max_values, mean_values, std_values], axis=0)

# Créez un nouveau DataFrame avec une seule ligne
summary_df = pd.DataFrame([summary_row], columns=summary_row.index)

# Affichez le DataFrame
print(summary_df)






GET_ROW=1
row = df_unique_spectrograms.iloc[GET_ROW]
df_spectrogram = pd.read_parquet(f'{train_spec_path}{row.spectrogram_id}.parquet')
spec_offset = int( row.spectrogram_label_offset_seconds )
df_spectrogram = df_spectrogram.loc[(df_spectrogram.time>=spec_offset)
                        &(df_spectrogram.time<spec_offset+600)]
    # Copy the Data
df_spectrogram = df_spectrogram.copy()
df_spectrogram = df_spectrogram.drop(['time'], axis=1)


# Calculez la moyenne de chaque colonne avec un préfixe 'mean_'
mean_values = df_spectrogram.mean()
mean_values = mean_values.add_prefix('mean_')

# Calculez le minimum de chaque colonne avec un préfixe 'min_'
min_values = df_spectrogram.min()
min_values = min_values.add_prefix('min_')

# Calculez le maximum de chaque colonne avec un préfixe 'max_'
max_values = df_spectrogram.max()
max_values = max_values.add_prefix('max_')

# Calculez l'écart type de chaque colonne avec un préfixe 'std_'
std_values = df_spectrogram.std()
std_values = std_values.add_prefix('std_')

# Concaténez les résultats pour obtenir une ligne
summary_row = pd.concat([min_values, max_values, mean_values, std_values], axis=0)

# Créez un nouveau DataFrame avec une seule ligne
summary_df1 = pd.DataFrame([summary_row], columns=summary_row.index)

# Affichez le DataFrame
print(summary_df1)



summary = pd.concat([summary_df,summary_df1],axis=0).reset_index()
print(summary)



'''
# Créez un DataFrame vide pour stocker les résumés
summary_df = pd.DataFrame()

# Définissez le nombre de lignes à traiter
num_rows_to_process = df_unique_spectrograms.shape[0]

# Boucle à travers les lignes
for GET_ROW in tqdm(range(num_rows_to_process)):
    # Votre code pour obtenir une ligne résumée pour chaque GET_ROW
    row = df_unique_spectrograms.iloc[GET_ROW]
    df_spectrogram = pd.read_parquet(f'{train_spec_path}{row.spectrogram_id}.parquet')
    spec_offset = int(row.spectrogram_label_offset_seconds)
    df_spectrogram = df_spectrogram.loc[(df_spectrogram.time >= spec_offset)
                                         & (df_spectrogram.time < spec_offset + 600)]
    df_spectrogram = df_spectrogram.copy()
    df_spectrogram = df_spectrogram.drop(['time'], axis=1)

    mean_values = df_spectrogram.mean()
    mean_values = mean_values.add_prefix('mean_')
    min_values = df_spectrogram.min()
    min_values = min_values.add_prefix('min_')
    max_values = df_spectrogram.max()
    max_values = max_values.add_prefix('max_')
    std_values = df_spectrogram.std()
    std_values = std_values.add_prefix('std_')

    summary_row = pd.concat([min_values, max_values, mean_values, std_values], axis=0)
    summary_df_row = pd.DataFrame([summary_row], columns=summary_row.index)

    # Ajoutez la ligne résumée au DataFrame principal
    summary_df = pd.concat([summary_df, summary_df_row], axis=0)

# Réinitialisez les index du DataFrame résultant
summary_df = summary_df.reset_index(drop=True)

# Sauvegardez le DataFrame résumé dans un fichier CSV
summary_df.to_csv('summary_dataframe.csv', index=False)
'''

summary_df = pd.read_csv('/Users/jordanmoles/Documents/Programmes_Informatiques/Python/Projects/Kaggle_Competitions/hms-harmful-brain-activity-classification/summary_dataframe.csv')

# Affichez le DataFrame résultant
print(summary_df)


data_spec = pd.concat([df_unique_spectrograms,summary_df], axis=1)


data_spec = data_spec.drop(['eeg_sub_id','eeg_label_offset_seconds','spectrogram_sub_id','spectrogram_label_offset_seconds','label_id', 'seizure_vote',  'lpd_vote',  'gpd_vote',  'lrda_vote',  'grda_vote', 'other_vote'], axis=1)
print(data_spec.head())


TARGETS = df.columns[-6:]


##############################################################################################
#                                           TRAIN TEST 
##############################################################################################


'''
# Create the train_set and the test_set
train_set, test_set = train_test_split(data_spec, test_size=0.2, random_state = 0)

# Display the quantity of negative and positive values in the 'SARS-Cov-2 exam result' column of the train_set
print(train_set['expert_consensus'].value_counts())

# Display the quantity of negative and positive values in the 'SARS-Cov-2 exam result' column of the test_set
print(test_set['expert_consensus'].value_counts())
'''



##############################################################################################
#                                           ENCODING
##############################################################################################


# Create an encoding function
def encoder(data_spec):
    code = {'Seizure':0,
            'LPD':1,
            'GPD':2,
            'LRDA':3,
            'GRDA':4,
            'Other':5}
    # Encode the train_set
    for col in data_spec.select_dtypes('object').columns:
        data_spec.loc[:,col] = data_spec[col].map(code)

    return data_spec
'''
# Display the new data frame
print(encoder(data_spec).head())

# Are there object type variables ? 
print(encoder(data_spec).dtypes.value_counts())

# Create a preprocessing function
def preprocessing(data_spec):
    data_spec = encoder(data_spec)

    X = data_spec.drop('expert_consensus', axis=1)
    y = data_spec['expert_consensus']

    print(y.value_counts())
    return X, y


# Create the X_train, y_train and the X_test, y_test
X_train, y_train = preprocessing(train_set)
X_test, y_test = preprocessing(test_set)
'''



'''
##############################################################################################
#                       INITIAL MODELING AND EVALUATION PROCEDURE
##############################################################################################


# A simple way of testing is with decision trees

# Create a decision tree model
model = DecisionTreeClassifier(random_state=0)

# Create evaluation function
def evaluation(model):
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    # Create learning curves
    N, train_score, val_score = learning_curve(model, X_train, y_train, cv = 4, scoring = 'accuracy', train_sizes=np.linspace(0.1,1,10))
    plt.figure(figsize=(12,8))
    plt.plot(N, train_score.mean(axis=1), label='train_score', c='blue')
    plt.plot(N, val_score.mean(axis=1), label='val_score', c='orange')
    plt.grid(ls='--')
    plt.legend()
    plt.show()
'''

'''
# Overfitting !!!
evaluation(model)


# Get feature importances
feature_importances = model.feature_importances_

# Get indices of top 50 features
top_indices = np.argsort(feature_importances)[-50:]

# Plot the top 50 features
plt.figure(figsize=(12, 8))
plt.barh(range(len(top_indices)), feature_importances[top_indices])
plt.yticks(range(len(top_indices)), X_train.columns[top_indices])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Top 50 Feature Importances')
plt.show()
'''






##############################################################################################
#                       SECOND MODELING AND EVALUATION PROCEDURE
##############################################################################################


'''
# Create a decision tree model
model = CatBoostClassifier(loss_function='MultiClass')




model.fit(X_train, y_train)


y_pred = model.predict(X_test)

# Create learning curves
N, train_score, val_score = learning_curve(model, X_train, y_train, cv = 4, scoring = 'accuracy', train_sizes=np.linspace(0.1,1,10))
plt.figure(figsize=(12,8))
plt.plot(N, train_score.mean(axis=1), label='train_score', c='blue')
plt.plot(N, val_score.mean(axis=1), label='val_score', c='orange')
plt.grid(ls='--')
plt.legend()
plt.show()

# Save the trained CatBoost model
model.save_model(f'CAT_v1_full_model.cat')



# Get feature importances
feature_importances = model.feature_importances_

# Get indices of top 50 features
top_indices = np.argsort(feature_importances)[-50:]

# Plot the top 50 features
plt.figure(figsize=(12, 8))
plt.barh(range(len(top_indices)), feature_importances[top_indices])
plt.yticks(range(len(top_indices)), X_train.columns[top_indices])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Top 50 Feature Importances')
plt.show()
'''