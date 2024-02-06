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


# Count the number of unique eeg_id, spectrogram_id, label_id, patient_id and Select columns with float type and observe the histogram
list_column_id = ['eeg_id', 'spectrogram_id', 'label_id', 'patient_id']
for id in list_column_id:
    df_count_id = df[id].value_counts().reset_index()
    df_count_id.columns = [id,'Count']
    print(df_count_id)


print(df_count_id.loc[df_count_id['Count'].idxmax(),'patient_id'])

print(df[df['patient_id']==df_count_id.loc[df_count_id['Count'].idxmax(),'patient_id']])

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


print(df['eeg_label_offset_seconds'].unique().sort_values())


