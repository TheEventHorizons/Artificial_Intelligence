import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns





data_train_path = '/Users/jordanmoles/Documents/Programmes_Informatiques/Python/Projects/Kaggle_Competitions/hms-harmful-brain-activity-classification/train.csv'



'''
############################################################################################################################################################################################
#                                                                                           CHECKLIST
############################################################################################################################################################################################

# Initial Form Analysis:

- Target Variable: (seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote)
- Rows and Columns: (106800, 15)
- Types of Variables: 12 int64, 2 float64, 1 object
- Analysis of Missing Variables: No missing value

# Initial Background Analysis:

- Target Visualization: The target variable is well balanced. Number of expert consensus for each type of activity is between 15000 and 20000
- Significance of Variables:
        * label_id: Nothing to say since it's a unique number representing each cases in the dataset
        * eeg_id: There are 17089 different eeg_id. Each eeg_id contains eeg_sub_id which can go from 0 to 742 associated to a eeg_label_offset_seconds which goes from 0.0 to 3372.0.
          Thus the occurency of a eeg_id goes from 1 to 743. A eeg_id is associated to a unique spectrogram_id.
        * spectrogram_id: There are 11138 different spectrogram_id Each spectrogram_id contains spectrogram_sub_id which can go from 0 to 1021 associated to a spectrogram_label_offset_seconds which goes from 0.0 to 17632.0?.
          Thus the occurency of a spectrogram_id goes from 1 to 1022. A spectrogram_id can contain different eeg_id  
        * patient_id: There are 1950 different patient_id. The occurrence is between 1 and 2215 corresponding to different combination of 
          ('eeg_id','eeg_sub_id', 'eeg_label_offset_seconds',spectrogram_id',spectrogram_sub_id',spectrogram_label_offset_seconds).       
- Relationship Target/target: it seems that seizure is correlated with grda_vote (24%), other_vote (21%), lrda_vote (17%), gpd_vote (13%), lpd_vote (13%). lrda_vote and gpd_vote (15%). The rest seems to be negligeable (-8%)
- Relationship Variables/Target: 
    * patient_id/expert_consensus: Each patient_id can have more than one expert_consensus corresponding to a particular configuration (seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote).
    * eeg_id, eeg_sub_id, eeg_label_offset_seconds/expert_consensus: Each eeg_id can have different expert consensus based on the eeg_sub_id or eeg_label_offset_seconds we look at.
    * spectrogram_id, spectrogram_sub_id, spectrogram_label_offset_seconds/expert_consensus: Each spectrogram_id can have different expert consensus based on spectrogram_sub_id, spectrogram_label_offset_seconds 
      but also the eeg_id we look at.
- Relationship Variables/variables: 
    * eeg_id/spectrogram_id: A eeg_id is associated to a unique spectrogram_id but a spectrogram_id can contain different eeg_id.
    * eeg_id/patient_id and spectrogram_id/patient_id: Each patient can have several different eeg_id and spectrogram_id.

'''





############################################################################################################################################################################################
#                                                                                           FORM ANALYSIS TRAIN
############################################################################################################################################################################################


# Display the max row and the max columns
pd.set_option('display.max_row',111)
#pd.set_option('display.max_columns',111)

# Read the data
data = pd.read_csv(data_train_path)

# Copy the Data
df = data.copy()

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








############################################################################################################################################################################################
#                                                                                           BACKGROUND ANALYSIS TRAIN
############################################################################################################################################################################################


##########################################
########## Target vizualisation ##########
##########################################

'''
# Define the target columns
target_columns = ['seizure_vote', 'lpd_vote','gpd_vote','lrda_vote','grda_vote','other_vote']

# expert_consensus column is a part of target column since it represent the max of votes
consensus_column = ['expert_consensus']

# Display few line of the target columns
print(df[target_columns].head())


# Display the proportion of types in the column 'expert_consensus'
count_df_consensus = df['expert_consensus'].value_counts().reset_index()
count_df_consensus.columns=['expert_consensus','count']

# Define colors
colors = sns.color_palette("viridis", len(count_df_consensus))

# Display df_consensus in a bar
plt.figure(figsize=(12,8))
plt.bar(count_df_consensus['expert_consensus'], count_df_consensus['count'], color=colors, zorder=2)
plt.xlabel('Expert consensus')
plt.ylabel('Count')
plt.title('Distribution of Expert Consensus')
plt.legend()
plt.grid(ls='--')
plt.show()
'''









##########################################
######  Significance of Variables  #######
##########################################


# label_id: Nothing to say since it's a unique number representing each cases in the dataset




# eeg_id, eeg_sub_id, eeg_label_offset_seconds: 

# There are 17089 different eeg_id. 
# Each eeg_id contains eeg_sub_id which can go from 0 to 742 associated to a eeg_label_offset_seconds which goes from 0.0 to 3372.0. Thus the occurency of a eeg_id goes from 1 to 743.
# A eeg_id is associated to a unique spectrogram_id.

'''
# Count the number of unique eeg_id
df_count_eeg_id = df['eeg_id'].value_counts().reset_index()
df_count_eeg_id.columns = ['eeg_id','Count']
print(df_count_eeg_id)


# Display if in a figure
plt.figure(figsize=(12,8))
plt.bar(df_count_eeg_id.index, df_count_eeg_id['Count'], color=colors, zorder=2)
plt.xlabel('eeg_id')
plt.ylabel('Count')
plt.title('Distribution of eeg_id')
plt.legend()
plt.grid(ls='--')
plt.show()


# Display the dataframe corresponding to the eeg_id which appears the most 'eeg_id'=2259539799
df_max_eeg_id = df[df['eeg_id']==df_count_eeg_id.loc[df_count_eeg_id['Count'].idxmax(),'eeg_id']]
print(df_max_eeg_id)

# Display the different result of expert consensus for this eeg_id
print(df_max_eeg_id['expert_consensus'].unique())

# Display the dataframe corresponding to the eeg_id which appears the least 'eeg_id'=98046913 (one of them)
df_min_eeg_id = df[df['eeg_id']==df_count_eeg_id.loc[df_count_eeg_id['Count'].idxmin(),'eeg_id']]
print(df_min_eeg_id)

# Display the different result of expert consensus for this eeg_id
print(df_min_eeg_id['expert_consensus'].unique())

# Display the min-max eeg_label_offset_seconds
print(df['eeg_label_offset_seconds'].min())
print(df['eeg_label_offset_seconds'].max())


# Count the unique spectrogram_ids for each eeg_id
eeg_spectrogram_count = df.groupby('eeg_id')['spectrogram_id'].nunique()

# Filter the eeg_ids that have more than one unique spectrogram_id
eeg_ids_with_multiple_spectrogram = eeg_spectrogram_count[eeg_spectrogram_count > 1]

# Display the result
print(eeg_ids_with_multiple_spectrogram)
'''






# spectrogram_id, spectrogram_sub_id, spectrogram_label_offset_seconds: 

# There are 11138 different spectrogram_id
# Each spectrogram_id contains spectrogram_sub_id which can go from 0 to 1021 associated to a spectrogram_label_offset_seconds which goes from 0.0 to 17632.0?. Thus the occurency of a spectrogram_id goes from 1 to 1022.
# A spectrogram_id can contain different eeg_id


'''
# Count the number of unique spectrogram_id
df_count_spectrogram_id = df['spectrogram_id'].value_counts().reset_index()
df_count_spectrogram_id.columns = ['spectrogram_id','Count']
print(df_count_spectrogram_id)


# Display if in a figure
plt.figure(figsize=(12,8))
plt.bar(df_count_spectrogram_id.index, df_count_spectrogram_id['Count'], color=colors, zorder=2)
plt.xlabel('spectrogram_id')
plt.ylabel('Count')
plt.title('Distribution of spectrogram_id')
plt.legend()
plt.grid(ls='--')
plt.show()


# Display the dataframe corresponding to the spectrogram_id which appears the most 'spectrogram_id'=2259539799
df_max_spectrogram_id = df[df['spectrogram_id']==df_count_spectrogram_id.loc[df_count_spectrogram_id['Count'].idxmax(),'spectrogram_id']]
print(df_max_spectrogram_id)

# Display the different result of expert consensus for this spectrogram_id
print(df_max_spectrogram_id['expert_consensus'].unique())

# Display the dataframe corresponding to the spectrogram_id which appears the least 'spectrogram_id'=98046913 (one of them)
df_min_spectrogram_id = df[df['spectrogram_id']==df_count_spectrogram_id.loc[df_count_spectrogram_id['Count'].idxmin(),'spectrogram_id']]
print(df_min_spectrogram_id)

# Display the different result of expert consensus for this spectrogram_id
print(df_min_spectrogram_id['expert_consensus'].unique())

# Display the min-max spectrogram_label_offset_seconds
print(df['spectrogram_label_offset_seconds'].min())
print(df['spectrogram_label_offset_seconds'].max())

'''



# patient_id: There are 1950 different patient_id. 

# The occurrence is between 1 and 2215 corresponding to different combination of ('eeg_id','eeg_sub_id', 'eeg_label_offset_seconds',spectrogram_id',spectrogram_sub_id',spectrogram_label_offset_seconds).


'''
# Count the number of unique Patient_id
df_count_patient_id = df['patient_id'].value_counts().reset_index()
df_count_patient_id.columns = ['patient_id','Count']
print(df_count_patient_id)
'''

'''
# Display if in a figure
plt.figure(figsize=(12,8))
plt.bar(df_count_patient_id.index, df_count_patient_id['Count'], color=colors, zorder=2)
plt.xlabel('patient_id')
plt.ylabel('Count')
plt.title('Distribution of patient_id')
plt.legend()
plt.grid(ls='--')
plt.show()
'''

'''
# Display the dataframe corresponding to the patient_id which appears the most ('patient_id'=30631)
df_max_patient_id = df[df['patient_id']==df_count_patient_id.loc[df_count_patient_id['Count'].idxmax(),'patient_id']]
print(df_max_patient_id)


# Display the dataframe corresponding to the patient_id which appears the least ('patient_id'=10324)
df_min_patient_id = df[df['patient_id']==df_count_patient_id.loc[df_count_patient_id['Count'].idxmin(),'patient_id']]
print(df_min_patient_id)
'''




##########################################
######  Relationship Target/Target  ######
##########################################

'''
# Define a correlation matrix between target columns
correlation_matrix_target = df[target_columns].corr()

# Plot the correlation matrix with clusters
plt.figure(figsize=(12,8))
sns.heatmap(correlation_matrix_target, annot=True, cmap='viridis')
plt.title('Correlation matrix between target columns')
plt.show()
'''





##########################################
######  Relationship Variable/Target  ####
##########################################



# patient_id/expert_consensus: Each patient_id can have more than one expert_consensus
'''

# Display the dataframe corresponding to columns 'patient_id', 'expert_consensus'
df_patient_id_expert_consensus = df[['patient_id','expert_consensus']]
print(df_patient_id_expert_consensus)

# Count the unique expert_consensus for each patient_id
patient_consensus_count = df.groupby('patient_id')['expert_consensus'].nunique()

# Filter the eeg_ids that have more than one unique spectrogram_id
patient_ids_with_multiple_expert_consensus = patient_consensus_count[patient_consensus_count > 1]

# Display the result
print(patient_ids_with_multiple_expert_consensus)

# Example of a patient_id with diffent expert_consensus
print(df[df['patient_id']==105])


# Créer un dataframe avec les colonnes 'patient_id' et 'expert_consensus'
df_patient_id_expert_consensus = df[['patient_id', 'expert_consensus']]

# Compter le nombre d'expert_consensus uniques pour chaque patient_id
patient_consensus_count = df.groupby('patient_id')['expert_consensus'].nunique()

# Filtrer les patient_ids qui ont plus d'un expert_consensus unique
patient_ids_with_multiple_expert_consensus = patient_consensus_count[patient_consensus_count > 1]

# Calculer le pourcentage de patients pour chaque nombre d'expert_consensus unique
percentage_dict = {}
for i in range(1, 6):
    count = (patient_consensus_count == i).sum()
    percentage = (count / len(patient_consensus_count)) * 100
    percentage_dict[i] = percentage

# Créer un plot à barres
plt.bar(percentage_dict.keys(), percentage_dict.values(), color='blue')
plt.xlabel('Nombre d\'expert_consensus uniques')
plt.ylabel('Pourcentage de patients')
plt.title('Pourcentage de patients avec différents nombres d\'expert_consensus')
plt.show()
'''




# eeg_id, eeg_sub_id, eeg_label_offset_seconds/expert_consensus: Each eeg_id can have different expert consensus based on the eeg_sub_id or eeg_label_offset_seconds we look at.

'''
# Display the dataframe corresponding to columns 'eeg_id', 'eeg_sub_id', 'eeg_label_offset_seconds', 'expert_consensus'
df_eeg_expert_consensus = df[['eeg_id','eeg_sub_id','eeg_label_offset_seconds','expert_consensus']]
print(df_eeg_expert_consensus)

# Count the unique expert_consensus for each eeg_id
eeg_consensus_count = df.groupby(['eeg_id'])['expert_consensus'].nunique()

# Filter the eeg_ids that have more than one unique expert_consensus
eeg_ids_with_multiple_expert_consensus = eeg_consensus_count[eeg_consensus_count > 1]

# Display the result
print(eeg_ids_with_multiple_expert_consensus)


# Example of a eeg_id with diffent expert_consensus
print(df[df['eeg_id']==21379701])
'''




# spectrogram_id, spectrogram_sub_id, spectrogram_label_offset_seconds/expert_consensus: Each spectrogram_id can have different expert consensus based on spectrogram_sub_id, spectrogram_label_offset_seconds 
#                                                                                        but also the eeg_id we look at.
                                                                                        
'''
# Display the dataframe corresponding to columns 'spectrogram_id', 'spectrogram_sub_id', 'spectrogram_label_offset_seconds', 'expert_consensus'
df_spec_expert_consensus = df[['spectrogram_id','spectrogram_sub_id','spectrogram_label_offset_seconds','expert_consensus']]
print(df_spec_expert_consensus)

# Count the unique expert_consensus for each spectrogram
spec_consensus_count = df.groupby(['spectrogram_id'])['expert_consensus'].nunique()

# Filter the spectrogram_ids that have more than one unique expert_consensus
spec_ids_with_multiple_expert_consensus = spec_consensus_count[spec_consensus_count > 1]

# Display the result
print(spec_ids_with_multiple_expert_consensus)

# Example of a spectrogram_id with diffent expert_consensus
print(df[df['spectrogram_id']==12849827])
'''











##########################################
### Relationship Variables/Variables  ####
##########################################


# eeg_id/spectrogram_id: A eeg_id is associated to a unique spectrogram_id but a spectrogram_id can contain different eeg_id

'''
# Count the unique spectrogram_ids for each eeg_id
eeg_spectrogram_count = df.groupby('eeg_id')['spectrogram_id'].nunique()

# Filter the eeg_ids that have more than one unique spectrogram_id and display the result
eeg_ids_with_multiple_spectrogram = eeg_spectrogram_count[eeg_spectrogram_count > 1]
print(eeg_ids_with_multiple_spectrogram)


# Count the number of unique spectrogram_id
df_count_spectrogram_id = df['spectrogram_id'].value_counts().reset_index()
df_count_spectrogram_id.columns = ['spectrogram_id','Count']

# Display the dataframe corresponding to the spectrogram_id which appears the most 'spectrogram_id'=2259539799
df_max_spectrogram_id = df[df['spectrogram_id']==df_count_spectrogram_id.loc[df_count_spectrogram_id['Count'].idxmax(),'spectrogram_id']]
print(df_max_spectrogram_id)
'''


# eeg_id/patient_id and spectrogram_id/patient_id: Each patient can have several different eeg_id and spectrogram_id

'''
# Count the number of unique Patient_id
df_count_patient_id = df['patient_id'].value_counts().reset_index()
df_count_patient_id.columns = ['patient_id','Count']
print(df_count_patient_id)


# Display the dataframe corresponding to the patient_id which appears the most ('patient_id'=30631)
df_max_patient_id = df[df['patient_id']==df_count_patient_id.loc[df_count_patient_id['Count'].idxmax(),'patient_id']]
print(df_max_patient_id)
'''




































############################################################################################################################################################################################
#                                                                                            ANALYSIS TRAIN EEG
############################################################################################################################################################################################

'''
# Form Analysis:

- Rows and Columns: (10000, 20) + (0,1) to facilitate our comprehension, corresponding the 'eeg_label_offset_seconds' column. Each combination ('eeg_id', 'eeg_sub_id') corresponds to a 50 second long subsample 
                    starting at time 'eeg_label_offset_seconds' where 200 samples were taken each second. 
- Types of Variables: 20 float64
- Analysis of Missing Variables: No missing value

# Background Analysis:

- Significance of Variables:
    * Each column represents a measure done by a particular electrode placed on the head of the patient
- Relationship Variables/Target: 

- Relationship Variables/variables: 
    * Variables seem to be generally correlated according to the distance in a defined montage
    * Correlations between variables change according to the label_id

'''


# Analysis of one particular eeg_id (the first one 'eeg_id'=1628180742)

train_eeg_path = '/Users/jordanmoles/Documents/Programmes_Informatiques/Python/Projects/Kaggle_Competitions/hms-harmful-brain-activity-classification/train_eegs/'
train_spec_path = '/Users/jordanmoles/Documents/Programmes_Informatiques/Python/Projects/Kaggle_Competitions/hms-harmful-brain-activity-classification/train_spectrograms/'
train = pd.read_csv('/Users/jordanmoles/Documents/Programmes_Informatiques/Python/Projects/Kaggle_Competitions/hms-harmful-brain-activity-classification/train.csv')

GET_ROW = 1
row = train.iloc[GET_ROW]
df_eeg = pd.read_parquet(f'{train_eeg_path}{row.eeg_id}.parquet')
eeg_offset = int( row.eeg_label_offset_seconds )
df_eeg = df_eeg.iloc[eeg_offset*200:(eeg_offset+50)*200]

'''
df_spectrogram = pd.read_parquet(f'{train_spec_path}{row.spectrogram_id}.parquet')
spec_offset = int( row.spectrogram_label_offset_seconds )
df_spectrogram = df_spectrogram.loc[(df_spectrogram.time>=spec_offset)
                     &(df_spectrogram.time<spec_offset+600)]
'''

'''
# Count the number of offset_seconds for a unique eeg_id
print(df.iloc[GET_ROW]['eeg_id'])
print(df[df['eeg_id'] == df.iloc[GET_ROW]['eeg_id']]['eeg_label_offset_seconds'].unique())

# Print the unique offset_seconds for a specific row
print(df[df.reset_index()['index']==GET_ROW]['eeg_label_offset_seconds'])


# Copy the Data
df_eeg = df_eeg.copy()

# Create columns time with offset_seconds
sample_per_second = 200
df_eeg['seconds_with_offset'] = range(df_eeg.shape[0]) 
df_eeg['seconds_with_offset'] = df_eeg['seconds_with_offset']/sample_per_second +df[df.reset_index()['index']==GET_ROW]['eeg_label_offset_seconds'].unique()



# Observe few lines 
print(df_eeg.head())

# Shape of the data
#print('The shape of df is:', df_eeg.shape)

# Create columns
Column_name = list(df_eeg.columns)

# Number of NaN in each column
number_na = df_eeg.isna().sum()

# Type of Data and the number
types = df_eeg.dtypes
number_types = df_eeg.dtypes.value_counts()
#print(number_types)

# Create a resume table
df_eeg_resume = pd.DataFrame({'features': Column_name, 'Type': types, 'Number of NaN': number_na})
#print(df_eeg_resume)
'''

'''
# Display the 3 first rows for the columns Fp1 and observe that there are overlaps
plt.figure(figsize=(12, 8))
for i in range(3):
    GET_ROW = i
    row = train.iloc[GET_ROW]
    df_eeg = pd.read_parquet(f'{train_eeg_path}{row.eeg_id}.parquet')
    eeg_offset = int( row.eeg_label_offset_seconds )
    df_eeg = df_eeg.iloc[eeg_offset*200:(eeg_offset+50)*200] 
    df_eeg = df_eeg.copy()
    df_eeg['seconds_with_offset'] = range(df_eeg.shape[0]) 
    df_eeg['seconds_with_offset'] = df_eeg['seconds_with_offset']/sample_per_second +df[df.reset_index()['index']==GET_ROW]['eeg_label_offset_seconds'].unique()
    color = plt.cm.viridis(i/8.0)
    plt.plot(df_eeg['seconds_with_offset'], df_eeg['Fp1'], c='black')
    plt.fill_between(df_eeg['seconds_with_offset'], -250, 0, where=[(x >= df_eeg['seconds_with_offset'].min()) and (x <= df_eeg['seconds_with_offset'].max()) for x in df_eeg['seconds_with_offset']], color=color, alpha=0.7, label=f'row {i}')
    plt.axvline(df_eeg['seconds_with_offset'].min(), color=color, linestyle='-', linewidth=2)
    plt.axvline(df_eeg['seconds_with_offset'].max(), color=color, linestyle='-', linewidth=2)
plt.ylabel('Fp1')
plt.legend()
plt.grid(ls='--')    
plt.show()
'''

'''
# Display the full eeg_id for the columns Fp1 and the ten seconds that we need to predict and observe that there are overlaps
plt.figure(figsize=(12, 8))
for i in range(9):
    GET_ROW = i
    row = train.iloc[GET_ROW]
    df_eeg = pd.read_parquet(f'{train_eeg_path}{row.eeg_id}.parquet')
    eeg_offset = int( row.eeg_label_offset_seconds )
    df_eeg = df_eeg.iloc[eeg_offset*200:(eeg_offset+50)*200] 
    df_eeg = df_eeg.copy()
    df_eeg['seconds_with_offset'] = range(df_eeg.shape[0]) 
    df_eeg['seconds_with_offset'] = df_eeg['seconds_with_offset']/sample_per_second +df[df.reset_index()['index']==GET_ROW]['eeg_label_offset_seconds'].unique()
    color = plt.cm.viridis(i/8.0)
    plt.plot(df_eeg['seconds_with_offset'], df_eeg['Fp1'], c='black')
    plt.fill_between(df_eeg['seconds_with_offset'], -250, 0, where=[(x >= ((df_eeg['seconds_with_offset'].max()+df_eeg['seconds_with_offset'].min())/2-5)) and (x <= ((df_eeg['seconds_with_offset'].max()+df_eeg['seconds_with_offset'].min())/2+5)) for x in df_eeg['seconds_with_offset']], color=color, alpha=0.7, label=f'row {i}')
    plt.axvline((df_eeg['seconds_with_offset'].max()+df_eeg['seconds_with_offset'].min())/2-5, color=color, linestyle='-', linewidth=2)
    plt.axvline((df_eeg['seconds_with_offset'].max()+df_eeg['seconds_with_offset'].min())/2+5, color=color, linestyle='-', linewidth=2)
plt.ylabel('Fp1')
plt.legend()
plt.grid(ls='--')    
plt.show()
'''

'''
# Display all the columns of a particular eeg_id, eeg_sub_id  and the ten seconds that we need to predict
plt.figure(figsize=(12, 8))
for i, col in enumerate(df_eeg.columns[:-2]):
    plt.subplot(len(df_eeg.columns),1,i+1)
    plt.plot(df_eeg['seconds_with_offset'], df_eeg[col], c='black')
    #plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    plt.fill_between(df_eeg['seconds_with_offset'], df_eeg[col].min(), df_eeg[col].max(), where=[(x >= (df_eeg['seconds_with_offset'].min()+df_eeg['seconds_with_offset'].max())/2-5) and (x <= (df_eeg['seconds_with_offset'].min()+df_eeg['seconds_with_offset'].max())/2+5) for x in df_eeg['seconds_with_offset']], color='lightblue', alpha=0.7)
    plt.axvline((df_eeg['seconds_with_offset'].max()+df_eeg['seconds_with_offset'].min())/2-5, color='lightblue', linestyle='-', linewidth=2)
    plt.axvline((df_eeg['seconds_with_offset'].max()+df_eeg['seconds_with_offset'].min())/2+5, color='lightblue', linestyle='-', linewidth=2)
    plt.ylabel(col)
    plt.grid(ls='--')
plt.show()
'''



'''
# Relationship variables eeg/variables eeg 

# Define the dataframe of electrodes
df_electrode = df_eeg.drop(['EKG','seconds_with_offset'], axis=1)

# Plot the correlation matrix between electrodes
sns.heatmap(df_electrode[df_electrode.columns].corr(), annot=True, cmap='viridis')
plt.show()
'''










############################################################################################################################################################################################
#                                                                                            ANALYSIS TRAIN SPECTROGRAM
############################################################################################################################################################################################

'''
# Form Analysis:

- Rows and Columns: (300, 401). Each combination ('spectrograms_id', 'spectrograms_sub_id') corresponds to a 600 seconds (ten minutes) long subsample starting at time 'spectrograms_label_offset_seconds' 
                                where 1 sample was taken each 2 seconds. 
- Types of Variables: 400 float32, 1 int64
- Analysis of Missing Variables: No missing value

# Background Analysis:

- Significance of Variables:
    * Each column represents a measure done by a particular groupe of electrode LL, LP, RL, RP placed on a particular place of the head of the patient and a column time
    * Observe that different spectrograms_sub_id with the same id coincide for a large part 

- Relationship Variables/variables: 
    * Variables RL and RP with the same suffix number seem to be generally highly correlated (>88%) 
    * Correlations between variables change according to the label_id

'''


# Analysis of one particular spectrogram_id (the first one 'spectrogram_id'=353733 associated to the previous eeg_id)


train_eeg_path = '/Users/jordanmoles/Documents/Programmes_Informatiques/Python/Projects/Kaggle_Competitions/hms-harmful-brain-activity-classification/train_eegs/'
train_spec_path = '/Users/jordanmoles/Documents/Programmes_Informatiques/Python/Projects/Kaggle_Competitions/hms-harmful-brain-activity-classification/train_spectrograms/'
train = pd.read_csv('/Users/jordanmoles/Documents/Programmes_Informatiques/Python/Projects/Kaggle_Competitions/hms-harmful-brain-activity-classification/train.csv')

GET_ROW = 0
row = train.iloc[GET_ROW]

'''
df_eeg = pd.read_parquet(f'{train_eeg_path}{row.eeg_id}.parquet')
eeg_offset = int( row.eeg_label_offset_seconds )
df_eeg = df_eeg.iloc[eeg_offset*200:(eeg_offset+50)*200]
'''


df_spectrogram = pd.read_parquet(f'{train_spec_path}{row.spectrogram_id}.parquet')
spec_offset = int( row.spectrogram_label_offset_seconds )
df_spectrogram = df_spectrogram.loc[(df_spectrogram.time>=spec_offset)
                     &(df_spectrogram.time<spec_offset+600)]


# Count the number of offset_seconds for a unique eeg_id
print(df.iloc[GET_ROW]['spectrogram_id'])
print(df[df['spectrogram_id'] == df.iloc[GET_ROW]['spectrogram_id']]['spectrogram_label_offset_seconds'].unique())

# Print the unique offset_seconds for a specific row
print(df[df.reset_index()['index']==GET_ROW]['spectrogram_label_offset_seconds'])


# Copy the Data
df_spectrogram = df_spectrogram.copy()

# Create columns time with offset_seconds
sample_per_second = 0.5
df_spectrogram['seconds_with_offset'] = range(df_spectrogram.shape[0]) 
df_spectrogram['seconds_with_offset'] = df_spectrogram['seconds_with_offset']/sample_per_second +df[df.reset_index()['index']==GET_ROW]['spectrogram_label_offset_seconds'].unique()



# Observe few lines 
print(df_spectrogram.head())

# Shape of the data
print('The shape of df is:', df_spectrogram.shape)

# Create columns
Column_name = list(df_spectrogram.columns)

# Number of NaN in each column
number_na = df_spectrogram.isna().sum()

# Type of Data and the number
types = df_spectrogram.dtypes
number_types = df_spectrogram.dtypes.value_counts()
print(number_types)

# Create a resume table
df_spectrogram_resume = pd.DataFrame({'features': Column_name, 'Type': types, 'Number of NaN': number_na})
print(df_spectrogram_resume)


'''
# Display the 3 first rows for the columns LL_0.59 and observe that there are overlaps
plt.figure(figsize=(12, 8))
for i in range(3):
    GET_ROW = i
    row = train.iloc[GET_ROW]
    df_spectrogram = pd.read_parquet(f'{train_spec_path}{row.spectrogram_id}.parquet')
    spec_offset = int( row.spectrogram_label_offset_seconds )
    df_spectrogram = df_spectrogram.loc[(df_spectrogram.time>=spec_offset)
                     &(df_spectrogram.time<spec_offset+600)]
    df_spectrogram = df_spectrogram.copy()
    df_spectrogram['seconds_with_offset'] = range(df_spectrogram.shape[0]) 
    df_spectrogram['seconds_with_offset'] = df_spectrogram['seconds_with_offset']/sample_per_second +df[df.reset_index()['index']==GET_ROW]['spectrogram_label_offset_seconds'].unique()
    color = plt.cm.viridis(i/8.0)
    plt.plot(df_spectrogram['seconds_with_offset'], df_spectrogram['LL_0.59'], c='black')
    plt.fill_between(df_spectrogram['seconds_with_offset'], 0, 18, where=[(x >= df_spectrogram['seconds_with_offset'].min()) and (x <= df_spectrogram['seconds_with_offset'].max()) for x in df_spectrogram['seconds_with_offset']], color=color, alpha=0.7, label=f'row {i}')
    plt.axvline(df_spectrogram['seconds_with_offset'].min(), color=color, linestyle='-', linewidth=2)
    plt.axvline(df_spectrogram['seconds_with_offset'].max(), color=color, linestyle='-', linewidth=2)
plt.ylabel('Fp1')
plt.legend()
plt.grid(ls='--')    
plt.show()
'''


'''
# Display the full spectrogram_id for the columns LL_0.59 and the ten seconds that we need to predict and observe that there are overlaps
plt.figure(figsize=(12, 8))
for i in range(9):
    GET_ROW = i
    row = train.iloc[GET_ROW]
    df_spectrogram = pd.read_parquet(f'{train_spec_path}{row.spectrogram_id}.parquet')
    spec_offset = int( row.spectrogram_label_offset_seconds )
    df_spectrogram = df_spectrogram.loc[(df_spectrogram.time>=spec_offset)
                     &(df_spectrogram.time<spec_offset+600)]
    df_spectrogram = df_spectrogram.copy()
    df_spectrogram['seconds_with_offset'] = range(df_spectrogram.shape[0]) 
    df_spectrogram['seconds_with_offset'] = df_spectrogram['seconds_with_offset']/sample_per_second +df[df.reset_index()['index']==GET_ROW]['spectrogram_label_offset_seconds'].unique()
    color = plt.cm.viridis(i/8.0)
    plt.plot(df_spectrogram['seconds_with_offset'], df_spectrogram['LL_0.59'], c='black')
    plt.fill_between(df_spectrogram['seconds_with_offset'], 0, 18, where=[(x >= ((df_spectrogram['seconds_with_offset'].max()+df_spectrogram['seconds_with_offset'].min())/2-5)) and (x <= ((df_spectrogram['seconds_with_offset'].max()+df_spectrogram['seconds_with_offset'].min())/2+5)) for x in df_spectrogram['seconds_with_offset']], color=color, alpha=0.7, label=f'row {i}')
    plt.axvline((df_spectrogram['seconds_with_offset'].max()+df_spectrogram['seconds_with_offset'].min())/2-5, color=color, linestyle='-', linewidth=2)
    plt.axvline((df_spectrogram['seconds_with_offset'].max()+df_spectrogram['seconds_with_offset'].min())/2+5, color=color, linestyle='-', linewidth=2)
plt.ylabel('Fp1')
plt.legend()
plt.grid(ls='--')    
plt.show()
'''


'''
# Display the columns (LL_0.59, RL_0.59, LP_0.59, RP_0.59) of a particular spectrogram_id, spectrogram__sub_id  and the ten seconds that we need to predict
plt.figure(figsize=(12, 8))
for i, col in enumerate(['LL_0.59', 'RL_0.59', 'LP_0.59', 'RP_0.59']):
    plt.subplot(len(['LL_0.59', 'RL_0.59', 'LP_0.59', 'RP_0.59']),1,i+1)
    plt.plot(df_spectrogram['seconds_with_offset'], df_spectrogram[col], c='black')
    #plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    plt.fill_between(df_spectrogram['seconds_with_offset'], df_spectrogram[col].min(), df_spectrogram[col].max(), where=[(x >= (df_spectrogram['seconds_with_offset'].min()+df_spectrogram['seconds_with_offset'].max())/2-5) and (x <= (df_spectrogram['seconds_with_offset'].min()+df_spectrogram['seconds_with_offset'].max())/2+5) for x in df_spectrogram['seconds_with_offset']], color='lightblue', alpha=0.7)
    plt.axvline((df_spectrogram['seconds_with_offset'].max()+df_spectrogram['seconds_with_offset'].min())/2-5, color='lightblue', linestyle='-', linewidth=2)
    plt.axvline((df_spectrogram['seconds_with_offset'].max()+df_spectrogram['seconds_with_offset'].min())/2+5, color='lightblue', linestyle='-', linewidth=2)
    plt.ylabel(col)
    plt.grid(ls='--')
plt.show()
'''







# Relationship variables spectrogram/variables sprectrogram zoom

# Define the dataframe of montage
df_montage = df_spectrogram[['LL_0.78', 'RL_0.78', 'LP_0.78', 'RP_0.78']]

# Plot the correlation matrix between electrodes
sns.heatmap(df_montage[df_montage.columns].corr(), annot=True, cmap='viridis')
plt.show()


print(df_spectrogram.head())