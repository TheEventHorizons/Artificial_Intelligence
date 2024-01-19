import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import validation_curve


data_path = '/Users/jordanmoles/Documents/GitHub/Artificial_Intelligence/Covid-19_project/dataset.xlsx'



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




##############################################################################################
#                                      FORM ANALYSIS
##############################################################################################

# Display the max row and the max columns
pd.set_option('display.max_row',111)
#pd.set_option('display.max_columns',111)

# Read the data
data = pd.read_excel(data_path)

# Copy the Data
df = data.copy()

# Observe few lines 
#print(df.head())

# Shape of the data
#print('The shape of df is:', df.shape)

# Create columns
Column_name = list(df.columns)

# Number of NaN in each column
number_na = df.isna().sum()

# Type of Data and the number
types = df.dtypes
number_types = df.dtypes.value_counts()
#print(number_types)

# Create a resume table
df_resume = pd.DataFrame({'features': Column_name, 'Type': types, 'Number of NaN': number_na, })
#print(df_resume)

# Percentage of missing values in an incresing order
percent_na = (number_na/df.shape[0]).sort_values(ascending=True)
print(percent_na)

'''
# Display the dataframe
plt.figure(figsize=(20,10))
sns.heatmap(df.isna(), cbar=False)
plt.show()
'''


########## Removing useless columns ##########

# Remove data that have more that 90% of missing values
df = df[df.columns[number_na/df.shape[0] < 0.90]]
print(df)
print(df.shape)

# Remove Patient id column
df = df.drop('Patient ID', axis = 1)

# Observe few rows
#print(df.head())

'''
# Display the new dataframe
plt.figure(figsize=(15,10))
sns.heatmap(df.isna(), cbar=False)
plt.show()
'''






##############################################################################################
#                                      BACKGROUND ANALYSIS
##############################################################################################




########## Target vizualisation ##########

# Number of positive and negative cases
print(df['SARS-Cov-2 exam result'].value_counts())
print(df['SARS-Cov-2 exam result'].value_counts(normalize=True))




########## Significance of Variables ##########

'''
# Quantitative variables

# Select columns with float type and observe the histogram
for col in df.select_dtypes('float64'):
    plt.figure()
    sns.distplot(df[col])
    print(col)
plt.show()


# Patient age quantile 
plt.figure()
sns.distplot(df['Patient age quantile'], bins=20)
plt.show()


# Qualitative variables


# Check the category for each column
for col in df.select_dtypes('object'):
    print(f'{col :-<50} - {df[col].unique()}')

# Count the category for each column and display in a pie
for col in df.select_dtypes('object'):
    print(f'{col :-<50} - {df[col].value_counts()}')
    df[col].value_counts().plot.pie()
    plt.show()
'''



########### Relationship Variables/Target ###########

# Creating positive and negative subsets

# Positive subset
positive_df = df[df['SARS-Cov-2 exam result'] == 'positive']
print(positive_df.head())

# Negative subset
negative_df = df[df['SARS-Cov-2 exam result'] == 'negative']
print(negative_df.head())


# Creating Blood and Viral subsets

# Definition of the missing rate
missing_rate = df.isna().sum()/df.shape[0]

# Creation of the Blood columns
blood_columns = df.columns[(missing_rate < 0.9) & (missing_rate > 0.88)]

# Creation of the viral columns
viral_columns = df.columns[(missing_rate < 0.88) & (missing_rate > 0.75)]

'''
# Relation Target/Blood
for col in blood_columns:
    plt.figure()
    sns.distplot(positive_df[col], label='positive')
    sns.distplot(negative_df[col], label='negative')
    plt.legend()
    plt.show()



# Relation Target/Age
plt.figure()
sns.distplot(positive_df['Patient age quantile'], label='positive')
sns.distplot(negative_df['Patient age quantile'], label='negative')
plt.legend()
plt.show()

#Another way 
plt.figure()
sns.countplot(x='Patient age quantile', hue = 'SARS-Cov-2 exam result', data = df)
plt.legend()
plt.show()



# Relation Target/Viral

# crosstab for two caterogies
for col in viral_columns:
    plt.figure()
    sns.heatmap(pd.crosstab(df['SARS-Cov-2 exam result'], df[col]), annot=True, fmt='d')
    plt.legend()
    plt.show()
'''

##############################################################################################
#                                     SLIGHTLY ADVANCED ANALYSIS
##############################################################################################


######## Relationship Variables/Variables ########


# * Blood Level Relationships

'''
# Plot the
sns.pairplot(df[blood_columns])
plt.show()


# Plot the correlation matrix
sns.heatmap(df[blood_columns].corr())
plt.show()

# Plot the correlation matrix with clusters
sns.clustermap(df[blood_columns].corr())
plt.show()
'''


# * Relation age/blood

'''
for col in blood_columns:
    plt.figure()
    sns.lmplot(x='Patient age quantile', y = col, hue = 'SARS-Cov-2 exam result', data = df)
    plt.show()
'''

# Display the correlation matrix
print(df.corr()['Patient age quantile'].sort_values())






# * Relation Influenza/rapid test

'''
# Influenza A
plt.figure()
sns.heatmap(pd.crosstab(df['Influenza A'], df['Influenza A, rapid test']), annot=True, fmt='d')
plt.legend()
plt.show()

# Influenza B
plt.figure()
sns.heatmap(pd.crosstab(df['Influenza B'], df['Influenza B, rapid test']), annot=True, fmt='d')
plt.legend()
plt.show()
'''


# * Relations Viral blood

# Create sick variable (A patient is sick without having covid) without columns rapid Test
df['is sick'] = np.sum(df[viral_columns[:-2]] == 'detected', axis=1)>=1
print(df.head())

# Create a sick and non-sick dataset
sick_df = df[df['is sick'] == True]
print(sick_df.head())
non_sick_df = df[df['is sick'] == False]
print(non_sick_df.head())

'''
for col in blood_columns:
    plt.figure()
    sns.distplot(sick_df[col], label='is sick')
    sns.distplot(non_sick_df[col], label='is not sick')
    plt.legend()
    plt.show()
'''

# Create an Hospitalization function
def Hospitalization(df):
    if df['Patient addmited to regular ward (1=yes, 0=no)'] == 1:
        return 'regular ward'
    elif df['Patient addmited to semi-intensive unit (1=yes, 0=no)'] == 1:
        return 'semi-intensive unit'
    elif df['Patient addmited to intensive care unit (1=yes, 0=no)'] == 1:
        return 'intensive care unit'
    else:
        return 'Unknown'
    
# Add status column into the dataframe
df['status'] = df.apply(Hospitalization, axis=1)

# Print few lines of the new dataframe
print(df.head())

'''
# Display blood rate of patient in different status
for col in blood_columns:
    plt.figure()
    for cat in df['status'].unique():
        sns.distplot(df[df['status'] == cat][col], label=f'status {cat}')
    plt.legend()
    plt.show()
'''




######## NaN Analysis ########


# Test: delete missing values
print(df.dropna().count())

# Count value per blood columns of virus columns
print(df[blood_columns].count())
print(df[viral_columns[:-2]].count())

# state of our target without missing values per group
df1 = df[viral_columns[:-2]]
df1['covid']=df['SARS-Cov-2 exam result']
print(df1.dropna()['covid'].value_counts(normalize=True))

df2 = df[blood_columns]
df2['covid']=df['SARS-Cov-2 exam result']
print(df2.dropna()['covid'].value_counts(normalize=True))





# T-Test

from scipy.stats import ttest_ind

#for this test we need a balanced number of classes

# Number of negative and positive patient
print(negative_df.shape)
print(positive_df.shape)

# Use a sampling method to take the same number of variables
balanced_negative_df = negative_df.sample(positive_df.shape[0])

# Creation of the test function to reject or not H0
def t_test(col):
    alpha = 0.02
    stat, p = ttest_ind(balanced_negative_df[col].dropna(), positive_df[col].dropna())
    if p < alpha:
        return 'H0 is rejected'
    else:
        return 0

# We test the function for each blood columns
for col in blood_columns:
    print(f'{col:-<50} - {t_test(col)}')




