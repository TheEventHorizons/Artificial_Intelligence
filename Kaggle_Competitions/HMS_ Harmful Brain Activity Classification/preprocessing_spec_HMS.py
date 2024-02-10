import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm


from sklearn.model_selection import train_test_split



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

# Number of NaN in each column
number_na = df.isna().sum()

# Type of Data and the number
types = df.dtypes
number_types = df.dtypes.value_counts()
print(number_types)

# Create a resume table
df_resume = pd.DataFrame({'features': Column_name, 'Type': types, 'Number of NaN': number_na})
print(df_resume)
'''



############################################################################################################################################################################################
#                                                                                          TRAIN SET
############################################################################################################################################################################################

# Select unique spectrograms
df_unique_spectrograms = df.drop_duplicates(subset='spectrogram_id').reset_index(drop=True)
#print(df_unique_spectrograms.head())
#print(df_unique_spectrograms.shape)



train_spec_path = '/Users/jordanmoles/Documents/Programmes_Informatiques/Python/Projects/Kaggle_Competitions/hms-harmful-brain-activity-classification/train_spectrograms/'
'''

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

# Affichez le DataFrame résultant
print(summary_df)












#columns = 
#data_spec = np.random.randn()

'''
#for GET_ROW in range(df_unique_spectrograms.shape[0]):

for GET_ROW in range(2):
#GET_ROW=0
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
print(summary_row)'''