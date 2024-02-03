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

- Target Variable: SARS-CoV-2 exam result
- Rows and Columns: 5 rows and 111 columns
- Types of Variables: Qualitative (70), Quantitative (41)
- Analysis of Missing Variables:
    * A lot of NaN (more than 50 percent of variables have 90% NaN)
    * 2 groups of data: 76% Viral test, 89% blood test

# Background Analysis:

- Target Visualization:
    * 10% positive cases
    * 90% negative cases
- Significance of Variables:
    * Standardized continuous variables, skewed (asymmetrical), blood test
    * Patient age quantile: difficult to interpret this graph, clearly, these data have been processed, one might think of age groups 0-5 years, but it could just as well be a mathematical transformation.
      We cannot know because the person who provided this dataset does not specify it anywhere. However, it is not very important.
    * Qualitative Variable: binary (0,1), viral, Rhinovirus seems very high
- Relationship Variables/Target:
    * Target/Blood: Monocytes, platelets, leukocytes seem to be related to COVID-19. We need to test it.
    * Target/Age: Individuals of low age seem to be very little contaminated? Be cautious; we do not know the age, and we do not know the dataset's date (if it concerns children, we know they are as affected as adults).
      However, this variable will be interesting to compare with blood test results.
    * Target/Viral: Double diseases are very rare. Rhinovirus/Enterovirus - COVID negative? Hypothesis to test? But it is impossible for the region to have undergone an epidemic of this virus.
      Moreover, we can very well have two viruses at the same time. All of this has no connection with COVID.

# Initial Conclusion:

- A lot of missing data (at best, we keep 20% of the dataset).
- Two interesting data groups: Blood/Viral.
- Almost no discriminant variable to distinguish positive/negative cases, which allows us to say that it is not really appropriate to predict whether an individual has COVID based on these simple blood tests.
  But it's okay; we still need to continue the analysis to try to see what we can learn.
- We can identify interesting variables that are likely to play a non-negligible role (monocyte, etc.).

# Detailed Analysis

- Relationship Variables/Variables:
    * Blood_data/blood_data: Some variables are highly correlated, over 90%.
    * Blood_data/Age: Very weak correlation between age and blood levels.
    * Viral_data/viral_data: Influenza rapid test gives poor results; we may need to drop it.
    * Disease/blood_data Relationship: Blood levels between sick and COVID individuals are different.
    * Hospitalization/disease Relationship:
    * Hospitalization/blood Relationship: Interesting in case we want to predict which department a patient should go to.

- NaN Analysis: If we remove them
    * Viral: 1350 variables remain with a negative/positive ratio (92/8).
    * Blood: 600 variables remain with a negative/positive ratio (87/13).
    * For both, there are 90 variables left.

# Null Hypotheses (H0):

- Individuals with COVID-19 have significantly different leukocyte, monocyte, platelet levels.
    H0 = The mean levels are EQUAL in positive and negative individuals. Tests show that this hypothesis is rejected.
- Individuals with any disease have significantly different levels.
    H0 =

'''



