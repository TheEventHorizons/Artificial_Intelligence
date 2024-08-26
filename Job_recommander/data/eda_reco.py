import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import validation_curve


data_path_jobs = '/Users/jordanmoles/Documents/GitHub/Technical_test/data/jobs.csv'
data_path_feedbacks = '/Users/jordanmoles/Documents/GitHub/Technical_test/data/feedbacks.csv'
data_path_users = '/Users/jordanmoles/Documents/GitHub/Technical_test/data/users.csv'
data_path_users_history = '/Users/jordanmoles/Documents/GitHub/Technical_test/data/users_history.csv'
data_path_test_users = '/Users/jordanmoles/Documents/GitHub/Technical_test/data/test_users.csv'



'''
##################### Checklist: ######################

# Form Analysis:

- Target Variable: 
- Rows and Columns: 
- Types of Variables: 
- Analysis of Missing Variables:

# Background Analysis:

- Target Visualization:

- Significance of Variables:
 
- Relationship Variables/Target:


# Initial Conclusion:


# Detailed Analysis

- Relationship Variables/Variables:


- NaN Analysis: If we remove them


# Null Hypotheses (H0):



'''




##############################################################################################
#                                      FORM ANALYSIS
##############################################################################################

# Display the max row and the max columns
pd.set_option('display.max_row',111)
#pd.set_option('display.max_columns',111)

# Read the data
data = pd.read_csv(data_path_feedbacks)

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
df_resume = pd.DataFrame({'features': Column_name, 'Type': types, 'Number of NaN': number_na, })
print(df_resume)







