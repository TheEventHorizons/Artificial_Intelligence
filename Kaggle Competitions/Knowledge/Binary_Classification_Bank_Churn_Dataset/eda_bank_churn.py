import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import validation_curve


data_path = '/Users/jordanmoles/Documents/GitHub/Artificial_Intelligence/Kaggle_Competitions/Knowledge/Binary_Classification_Bank_Churn_Dataset/playground_series_s4e1/train.csv'



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


- NaN Analysis: 

# Null Hypotheses (H0):

- Individuals with COVID-19 have significantly different leukocyte, monocyte, platelet levels.
    H0 = The mean levels are EQUAL in positive and negative individuals. Tests show that this hypothesis is rejected.
- Individuals with any disease have significantly different levels.
    H0 =

'''




##############################################################################################
#                                      FORM ANALYSIS
##############################################################################################

# Display the max row and the max columns
pd.set_option('display.max_row',111)
#pd.set_option('display.max_columns',111)

# Read the data
data = pd.read_csv(data_path)

# Copy the Data
df = data.copy()

# Observe few lines 
print(df.head())





