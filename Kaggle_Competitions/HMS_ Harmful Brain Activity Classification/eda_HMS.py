import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import validation_curve


data_train_path = '/Users/jordanmoles/Documents/Programmes_Informatiques/Python/Projects/Kaggle_Competitions/hms-harmful-brain-activity-classification/train.csv'



'''
##################### Checklist: ######################

# Form Analysis:

- Target Variable: (seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote)
- Rows and Columns: (106800, 15)
- Types of Variables: 12 int64, 2 float64, 1 object
- Analysis of Missing Variables: No missing value

# Background Analysis:

- Target Visualization: The target variable is well balanced. Number of votes for each type of activity is between 15000 and 20000
- Significance of Variables:
- Relationship Variables/Target:


# Initial Conclusion:

# Detailed Analysis

- Relationship Variables/Variables:

- NaN Analysis: 

# Null Hypotheses (H0):


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



########## Target vizualisation ##########

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
'''
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







########## Significance of Variables ##########




# label_id: Nothing to say since it's a unique number representing each cases in the dataset





# eeg_id, eeg_sub_id, eeg_label_offset_seconds: 

# There are 17089 different eeg_id. 
# Each eeg_id contains eeg_sub_id which can go from 0 to 742 associated to a eeg_label_offset_seconds which goes from 0.0 to 3372.0. Thus the occurency of a eeg_id goes from 1 to 743.
# A eeg_id is associated to a unique spectrogram_id.


# Count the number of unique eeg_id
df_count_eeg_id = df['eeg_id'].value_counts().reset_index()
df_count_eeg_id.columns = ['eeg_id','Count']
print(df_count_eeg_id)

'''
# Display if in a figure
plt.figure(figsize=(12,8))
plt.bar(df_count_eeg_id.index, df_count_eeg_id['Count'], color=colors, zorder=2)
plt.xlabel('eeg_id')
plt.ylabel('Count')
plt.title('Distribution of eeg_id')
plt.legend()
plt.grid(ls='--')
plt.show()
'''

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


# Display the dataframe corresponding to the eeg_id which appears the most 'eeg_id'=2259539799
df_max_spectrogram_id = df[df['spectrogram_id']==df_count_spectrogram_id.loc[df_count_spectrogram_id['Count'].idxmax(),'spectrogram_id']]
print(df_max_spectrogram_id)

# Display the different result of expert consensus for this eeg_id
print(df_max_spectrogram_id['expert_consensus'].unique())

# Display the dataframe corresponding to the eeg_id which appears the least 'eeg_id'=98046913 (one of them)
df_min_spectrogram_id = df[df['spectrogram_id']==df_count_spectrogram_id.loc[df_count_spectrogram_id['Count'].idxmin(),'spectrogram_id']]
print(df_min_spectrogram_id)

# Display the different result of expert consensus for this eeg_id
print(df_min_spectrogram_id['expert_consensus'].unique())

# Display the min-max spectrogram_label_offset_seconds
print(df['spectrogram_label_offset_seconds'].min())
print(df['spectrogram_label_offset_seconds'].max())
'''




# patient_id: There are 1950 different patient_id. The occurrence is between 1 and 2215 corresponding to different combination of ('eeg_id','eeg_sub_id', 'eeg_label_offset_seconds',spectrogram_id',spectrogram_sub_id',spectrogram_label_offset_seconds).
#                                                  Some combinations appears few times
'''
# Count the number of unique Patient_id
df_count_patient_id = df['patient_id'].value_counts().reset_index()
df_count_patient_id.columns = ['patient_id','Count']
print(df_count_patient_id)

# Display if in a figure
plt.figure(figsize=(12,8))
plt.bar(df_count_patient_id.index, df_count_patient_id['Count'], color=colors, zorder=2)
plt.xlabel('patient_id')
plt.ylabel('Count')
plt.title('Distribution of patient_id')
plt.legend()
plt.grid(ls='--')
plt.show()

# Display the dataframe corresponding to the patient_id which appears the most ('patient_id'=30631)
df_max_patient_id = df[df['patient_id']==df_count_patient_id.loc[df_count_patient_id['Count'].idxmax(),'patient_id']]
print(df_max_patient_id)
'''






'''
# Count the number of unique eeg_id, spectrogram_id, label_id, patient_id and Select columns with float type and observe the histogram
list_column_id = ['eeg_id', 'spectrogram_id', 'label_id', 'patient_id']
for id in list_column_id:
    df_count_id = df[id].value_counts().reset_index()
    df_count_id.columns = [id,'Count']
    print(df_count_id)


print(df_count_id.loc[df_count_id['Count'].idxmax(),'patient_id'])

print(df[df['patient_id']==df_count_id.loc[df_count_id['Count'].idxmax(),'patient_id']])
'''

'''
plt.figure(figsize=(12,8))
for id in list_column_id:
    df_counts = df[id].value_counts().reset_index()
    df_counts.columns=[id,'count']
    plt.bar(df_counts.index, df_counts['count'], color=colors, zorder=2)
    plt.xlabel(id)
    plt.ylabel('Count')
    plt.title(f'Distribution of {id}')
    plt.legend()
    plt.grid(ls='--')
    plt.show()
'''


