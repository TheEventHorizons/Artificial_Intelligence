import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import validation_curve


data_path = '/Users/jordanmoles/Documents/Programmes_Informatiques/Python/Projects/Kaggle_Competitions/predict-energy-behavior-of-prosumers/train.csv'



'''
##################### Checklist: ######################

# Form Analysis:

- Target Variable: target and is_consumption
- Rows and Columns: (2018352, 9)
- Types of Variables: 7 int64, 1 float64
- Analysis of Missing Variables: 528 in the target column


# Background Analysis:

- Target Visualization: From 0.0 to 15480.274
- Significance of Variables:
    * Train variables : county (0 to 15), product_type (0 to 3), is_consumption and is_business binary variables, data_block_id (0 to 637), prediction_unit_id (0 to 68),
                        time (2021-09-01 to 2024-01_26)
                        For a day, we observe that consumption decrease from 6 or 7 am to 5pm  
    * Weather: 
        historical_weather
        forecast_weather
    * Electricity prices
    * Gas Prices
    * client
    
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
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

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





##############################################################################################
#                                      FORM ANALYSIS
##############################################################################################





########## Target vizualisation ##########
'''
# Range of target
print(df['target'].value_counts())
print(df['target'].max())
print(df['target'].min())

plt.figure()
sns.displot(df['target'])
print('target')
plt.show()


########## Significance of Variables ##########


# No quantitative variables expect target


# Qualitative variables

# Check the category for each column
for col in df.select_dtypes('int64'):
    print(f'{col :-<50} - {df[col].unique()}')

# Count the category for each column and display in a pie
for col in df.select_dtypes('int64'):
    print(f'{col :-<50} - {df[col].value_counts()}')
    df[col].value_counts().plot.pie()
    plt.show()
'''

# Creating consumption and production targets

# consumption subset
consumption_df = df[df['is_consumption'] == 1]
consumption_df = consumption_df.drop(['is_consumption', 'row_id'], axis=1)
consumption_df = consumption_df.rename(columns={'target':'target_consumption'})
print(consumption_df.head())

# production subset
production_df = df[df['is_consumption'] == 0]
production_df = production_df.rename(columns={'target':'target_production'})
production_df = production_df.drop(['is_consumption', 'row_id'], axis=1)
print(production_df.head())

production_df['target_consumption'] = consumption_df['target_consumption']

df = production_df
print(df.head())


# Target over times when county, is_business and product_type are fixed

df = df[(df['county'] == 0) & (df['is_business'] == 0) & (df['product_type'] == 1)]
print(df.head())

# Observe the production and a consumption over a day
selected_day = df.loc['2021-09-01 00:00:00':'2021-09-02 00:00:00']
plt.figure()
sns.lineplot(x=selected_day.index, y=selected_day['target_consumption'],  label = 'consumption')
sns.lineplot(x=selected_day.index, y=selected_day['target_production'],  label = 'production')
plt.grid(ls='--')
plt.ylabel('target')
plt.legend()
plt.show()

# Observe the production and a consumption over a week
selected_week = df.loc['2021-09-01 00:00:00':'2021-09-08 00:00:00']
plt.figure()
sns.lineplot(x=selected_week.index, y=selected_week['target_consumption'],  label = 'consumption')
sns.lineplot(x=selected_week.index, y=selected_week['target_production'],  label = 'production')
plt.grid(ls='--')
plt.ylabel('target')
plt.legend()
plt.show()

# Observe the production and a consumption over a year
selected_year = df.loc['2022-01-01 00:00:00':'2023-01-01 00:00:00']
plt.figure()
sns.lineplot(x=selected_year.index, y=selected_year['target_consumption'], label = 'consumption')
sns.lineplot(x=selected_year.index, y=selected_year['target_production'], label = 'production', alpha = 0.7)
plt.ylabel('target')
plt.grid(ls='--')
plt.legend()
plt.show()


