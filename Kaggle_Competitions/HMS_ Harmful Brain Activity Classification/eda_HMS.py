import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import validation_curve


data_path = '/Users/jordanmoles/Documents/Programmes_Informatiques/Python/Projects/Kaggle_Competitions/hms-harmful-brain-activity-classification/train.cvs'



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



