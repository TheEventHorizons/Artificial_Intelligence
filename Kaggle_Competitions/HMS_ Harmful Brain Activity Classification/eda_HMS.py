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

- Target Visualization:
- Significance of Variables:
- Relationship Variables/Target:


# Initial Conclusion:

# Detailed Analysis

- Relationship Variables/Variables:

- NaN Analysis: 

# Null Hypotheses (H0):


'''





##############################################################################################
#                                      FORM ANALYSIS
##############################################################################################

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


# Count the number of unique eeg_id, spectrogram_id, label_id, patient_id
list_column_id = ['eeg_id', 'spectrogram_id', 'label_id','patient_id']
for id in list_column_id:
    print(f'The number of unique element in the column {id} is ', len(df[id].unique()))




##############################################################################################
#                                      BACKGROUND ANALYSIS
##############################################################################################


target_columns = ['seizure_vote', 'lpd_vote','gpd_vote','lrda_vote','grda_vote','other_vote']

########## Target vizualisation ##########

print(df[target_columns].head())



########## Significance of Variables ##########



'''
plt.figure(figsize=(12,8))
sns.heatmap(df[target_columns], cbar=True)
plt.show()
'''

'''
# Select columns with float type and observe the histogram
plt.figure()
sns.displot(df['eeg_id'], bins=len(df['eeg_id'].unique()))
print('eeg_id')
plt.show()

# Select columns with float type and observe the histogram
plt.figure()
sns.displot(df['spectrogram_id'], bins=len(df['spectrogram_id'].unique()))
print('eeg_id')
plt.show()
'''