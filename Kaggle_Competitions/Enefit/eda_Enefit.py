import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.tseries.offsets import DateOffset
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import validation_curve


data_path_train = '/Users/jordanmoles/Documents/Programmes_Informatiques/Python/Projects/Kaggle_Competitions/predict-energy-behavior-of-prosumers/train.csv'
data_path_historical_weather = '/Users/jordanmoles/Documents/Programmes_Informatiques/Python/Projects/Kaggle_Competitions/predict-energy-behavior-of-prosumers/historical_weather.csv'
data_path_forecast_weather = '/Users/jordanmoles/Documents/Programmes_Informatiques/Python/Projects/Kaggle_Competitions/predict-energy-behavior-of-prosumers/forecast_weather.csv'
data_path_electricity_prices = '/Users/jordanmoles/Documents/Programmes_Informatiques/Python/Projects/Kaggle_Competitions/predict-energy-behavior-of-prosumers/electricity_prices.csv'
data_path_client = '/Users/jordanmoles/Documents/Programmes_Informatiques/Python/Projects/Kaggle_Competitions/predict-energy-behavior-of-prosumers/client.csv'
data_path_gas_prices = '/Users/jordanmoles/Documents/Programmes_Informatiques/Python/Projects/Kaggle_Competitions/predict-energy-behavior-of-prosumers/gas_prices.csv'
data_path_weather_station_to_county_mapping = '/Users/jordanmoles/Documents/Programmes_Informatiques/Python/Projects/Kaggle_Competitions/predict-energy-behavior-of-prosumers/weather_station_to_county_mapping.csv'



'''
##################### Checklist: ######################

# Form Analysis:

o Train
- Target Variable: target and is_consumption
- Rows and Columns: (2018352, 9)
- min-max datetime:
- Types of Variables: 7 int64, 1 float64
- Analysis of Missing Variables: 528 in the target column


o historical_weather
- Rows and Columns: (1710802, 18)
- min-max datetime: from 2021-09-01 00:00:00 to 2023-05-30 10:00:00
- Types of Variables: 5 int64, 12 float64
- Analysis of Missing Variables: No missing variables


o forecast_weather
- Rows and Columns: (3424512, 18)
- min-max datetime: from 2021-09-01 02:00:00 to 2023-05-30 02:00:00
- Types of Variables: 14 float64, 2 object, 2 int64
- Analysis of Missing Variables: 2 


o weather_station_to_county_mapping
- Rows and Columns: (112, 4)
- Types of Variables: 3 float64, 1 object
- Analysis of Missing Variables: 63 missing values in the county column corresponding to county_name column


o electricity_prices
- Rows and Columns: (15286, 4)
- Types of Variables: 2 object, 1 float64, 1 int64
- Analysis of Missing Variables: No missing value


o client
- Rows and Columns: (41919, 7)
- min-max datetime:
- Types of Variables: 5 int64, 1 float64, 1 object
- Analysis of Missing Variables: No missing values


o gas_prices
- Rows and Columns: (637, 5)
- min-max datetime:
- Types of Variables: 2 object, 1 float64, 1 int64 
- Analysis of Missing Variables: No missing values




# Background Analysis:

- Target Visualization: From 0.0 to 15480.274
- Significance of Variables:
    * Train variables : county (0 to 15), product_type (0 to 3), is_consumption and is_business binary variables, data_block_id (0 to 637), prediction_unit_id (0 to 68),
                        time (2021-09-01 00:00:00 to 2023-05-31 23:00:00)
                        For one day, significant production during the day and minimal activity at night. Additionally, there is a decline in consumption between 6 AM and 5 PM.
                        for one week, it seems that there is a decrease of production around day 06 and 08 (2021-09) maybe because of weather ?
                        for the year 2022, Production seems to increase during summer  (and is almost zero on winter) whereas consumption decrease during summer and increase during winter.
                        not all product types exist in all county. 
                            - when business is 0, product_type is 0 for each county, only 2 are 0 for product_type 1 (6 and 12), only 3 are non 0 for product_type 2 (0, 7 and 11), and only 2 are 0 for product_type 3 (6 and 12)
                            - when business is 1, product_type is non 0 only for  county (0, 4, 5, 7, 11 and 15) , only 4 are 0 for product_type 1 (1, 6, 8 and 12), only 3 are non 0 for product_type 2 (0, 10 and 11), 
                              and all counties are non 0 for product_type 3                
    * weather_station_to_county_mapping: we associate counties to longitude and latitude, some counties have more than one station
    * Weather: 
        historical_weather:
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




############################################################################################################################################################################################
#                                                                                       FORM ANALYSIS
############################################################################################################################################################################################




##############################################################################################
#                                      Train
##############################################################################################



'''
# Display the max row and the max columns
pd.set_option('display.max_row',111)
#pd.set_option('display.max_columns',111)

# Read the data
data = pd.read_csv(data_path_train, parse_dates=['datetime'], index_col='datetime')

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
df_resume = pd.DataFrame({'features': Column_name, 'Type': types, 'Number of NaN': number_na })
print(df_resume)

# Print max-min index
print(df.index.min())
print(df.index.max())
'''




##############################################################################################
#                                      weather_station_to_county_mapping
##############################################################################################




# Read the data
data_weather_station_to_county_mapping = pd.read_csv(data_path_weather_station_to_county_mapping)

# Copy the Data
df_weather_station_to_county_mapping = data_weather_station_to_county_mapping.copy()

# Observe few lines 
print(df_weather_station_to_county_mapping.head())


# Shape of the data
print('The shape of df is:', df_weather_station_to_county_mapping.shape)


# Create columns
Column_name = list(df_weather_station_to_county_mapping.columns)

# Number of NaN in each column
number_na = df_weather_station_to_county_mapping.isna().sum()

# Type of Data and the number
types = df_weather_station_to_county_mapping.dtypes
number_types = df_weather_station_to_county_mapping.dtypes.value_counts()
print(number_types)

# Create a resume table
df_weather_station_to_county_mapping_resume = pd.DataFrame({'features': Column_name, 'Type': types, 'Number of NaN': number_na })
#print(df_weather_station_to_county_mapping_resume)

df_weather_station_to_county_mapping['latitude'] = df_weather_station_to_county_mapping['latitude'].round(1)

# Print max-min index
#print(df_weather_station_to_county_mapping.index.min())
#print(df_weather_station_to_county_mapping.index.max())

# Remove useless rows
df_weather_station_to_county_mapping = df_weather_station_to_county_mapping.dropna(axis=0)
#print(df_weather_station_to_county_mapping.head())
#print(df_weather_station_to_county_mapping.value_counts().sort_values())





##############################################################################################
#                                      Historical_weather
##############################################################################################



# Read the data
data_historical_weather = pd.read_csv(data_path_historical_weather)

# Copy the Data
df_historical_weather = data_historical_weather.copy()

# Observe few lines 
print(df_historical_weather.head())


# Shape of the data
print('The shape of df is:', df_historical_weather.shape)


# Create columns
Column_name = list(df_historical_weather.columns)

# Number of NaN in each column
number_na = df_historical_weather.isna().sum()

# Type of Data and the number
types = df_historical_weather.dtypes
number_types = df_historical_weather.dtypes.value_counts()
print(number_types)

# Create a resume table
df_historical_weather_resume = pd.DataFrame({'features': Column_name, 'Type': types, 'Number of NaN': number_na })
#print(df_historical_weather_resume)

# Print max-min index
#print(df_historical_weather.index.min())
#print(df_historical_weather.index.max())

# add column county associated to longitude and latitude
df_historical_weather=pd.merge(df_historical_weather, df_weather_station_to_county_mapping[['latitude', 'longitude', 'county']], on=['latitude', 'longitude'], how='left')


print(df_historical_weather[df_historical_weather['county']==2.0])

# use datetime index
#df_historical_weather.set_index('datetime', inplace=True)

# remove useless rows
#df_historical_weather = df_historical_weather.dropna(axis=0)

# creating average historical characteristics for weather stations related to county
df_historical_weather = df_historical_weather.groupby(['datetime', 'county']).mean()

# drop latitude and longitude columns
df_historical_weather= df_historical_weather.drop(['latitude', 'longitude'], axis=1)

# historical_weather during one day
#print(df_historical_weather['2021-09-01 00:00:00':'2021-09-01 01:00:00'])
#print(df_historical_weather.shape)





##############################################################################################
#                                      Forecast_weather
##############################################################################################



# Read the data
data_forecast_weather = pd.read_csv(data_path_forecast_weather)

# Copy the Data
df_forecast_weather = data_forecast_weather.copy()

# Observe few lines 
print(df_forecast_weather.head())

'''
##
pd.set_option('display.max_row',111)
selected_rows = df_forecast_weather.loc[
    (df_forecast_weather['datetime'] >= '2021-09-01 00:00:00') &
    (df_forecast_weather['datetime'] <= '2021-09-02 00:00:00')
]
print(selected_rows)

print(df_forecast_weather[df_forecast_weather['datetime'] == '2021-09-01 00:00:00'])
##
'''

# Shape of the data
print('The shape of df is:', df_forecast_weather.shape)


# Create columns
Column_name = list(df_forecast_weather.columns)

# Number of NaN in each column
number_na = df_forecast_weather.isna().sum()

# Type of Data and the number
types = df_forecast_weather.dtypes
number_types = df_forecast_weather.dtypes.value_counts()
print(number_types)

# Create a resume table
df_forecast_weather_resume = pd.DataFrame({'features': Column_name, 'Type': types, 'Number of NaN': number_na })
print(df_forecast_weather_resume)

# Print max-min index
print(df_forecast_weather.index.min())
print(df_forecast_weather.index.max())

# Extract and drop some columns
columns_forecast = df_forecast_weather[['forecast_datetime', 'hours_ahead']]
df_forecast_weather = df_forecast_weather.drop(['forecast_datetime','hours_ahead'], axis=1)

# add column county associated to longitude and latitude
df_forecast_weather=pd.merge(df_forecast_weather, df_weather_station_to_county_mapping[['latitude', 'longitude', 'county']], on=['latitude', 'longitude'], how='left')


print(df_forecast_weather[df_forecast_weather['county']==1.0])

# use datetime index
#df_forecast_weather.set_index('datetime', inplace=True)

# remove useless rows
#df_forecast_weather = df_forecast_weather.dropna(axis=0)

# creating average historical characteristics for weather stations related to county
#df_forecast_weather = df_forecast_weather.groupby(['datetime', 'county']).mean()

# drop latitude and longitude columns
#df_forecast_weather= df_forecast_weather.drop(['latitude', 'longitude'], axis=1)

# historical_weather during one day
#print(df_forecast_weather['2021-09-01 00:00:00':'2021-09-01 01:00:00'])
#print(df_forecast_weather.shape)









##############################################################################################
#                                      electricity_prices
##############################################################################################


'''
# Read the data
data_electricity_prices = pd.read_csv(data_path_electricity_prices)

# Copy the Data
df_electricity_prices = data_electricity_prices.copy()

# Observe few lines 
print(df_electricity_prices.head())



# Shape of the data
print('The shape of df is:', df_electricity_prices.shape)


# Create columns
Column_name = list(df_electricity_prices.columns)

# Number of NaN in each column
number_na = df_electricity_prices.isna().sum()

# Type of Data and the number
types = df_electricity_prices.dtypes
number_types = df_electricity_prices.dtypes.value_counts()
print(number_types)

# Create a resume table
df_electricity_prices_resume = pd.DataFrame({'features': Column_name, 'Type': types, 'Number of NaN': number_na })
print(df_electricity_prices_resume)
'''




##############################################################################################
#                                           client
##############################################################################################

'''
# Read the data
data_client = pd.read_csv(data_path_client)

# Copy the Data
df_client = data_client.copy()

# Observe few lines 
print(df_client.head())



# Shape of the data
print('The shape of df is:', df_client.shape)


# Create columns
Column_name = list(df_client.columns)

# Number of NaN in each column
number_na = df_client.isna().sum()

# Type of Data and the number
types = df_client.dtypes
number_types = df_client.dtypes.value_counts()
print(number_types)

# Create a resume table
df_client_resume = pd.DataFrame({'features': Column_name, 'Type': types, 'Number of NaN': number_na })
print(df_client_resume)
'''



##############################################################################################
#                                         gas_prices
##############################################################################################




'''
# Read the data
data_gas_prices = pd.read_csv(data_path_gas_prices)

# Copy the Data
df_gas_prices = data_gas_prices.copy()

# Observe few lines 
print(df_gas_prices.head())



# Shape of the data
print('The shape of df is:', df_gas_prices.shape)


# Create columns
Column_name = list(df_gas_prices.columns)

# Number of NaN in each column
number_na = df_gas_prices.isna().sum()

# Type of Data and the number
types = df_gas_prices.dtypes
number_types = df_gas_prices.dtypes.value_counts()
print(number_types)

# Create a resume table
df_gas_prices_resume = pd.DataFrame({'features': Column_name, 'Type': types, 'Number of NaN': number_na })
print(df_gas_prices_resume)
'''












































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

# Time

# min-max index time 
print('Min index is:',  df.index.min())
print('Max index is:', df.index.max())

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
'''
'''
# Target over times when county, is_business and product_type are fixed

df = df[(df['county'] == 0) & (df['is_business'] == 0) & (df['product_type'] == 3)]
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
'''


# Plot each combination of target, county, is_business and product_type

'''
for a in df['is_business'].unique():
    for i in df['county'].unique():
         for j in df['product_type'].unique():
            selected_year = df[(df['county'] == i) & (df['is_business'] == a) & (df['product_type'] == j)].loc['2021-09-01 00:00:00':'2023-05-31 23:00:00']
            plt.figure()
            sns.lineplot(x=selected_year.index, y=selected_year['target_consumption'], label = 'consumption')
            sns.lineplot(x=selected_year.index, y=selected_year['target_production'], label = 'production', alpha = 0.7)
            plt.ylabel('target')
            plt.title(f'county = {i}, product_type = {j} and is_business {a}')
            plt.grid(ls='--')
            plt.legend()
            plt.show()


# Count the number of product types in counties according to business and store it
counties = df["county"].unique()
product_types = [0, 1, 2, 3]

# Initialize an empty matrix for is_business = 0
matrix_business_0 = np.zeros((len(counties), len(product_types)))

# Initialize an empty matrix for is_business = 1
matrix_business_1 = np.zeros((len(counties), len(product_types)))

for i, product_type in enumerate(product_types):
    for j, county in enumerate(counties):
        # Count for is_business = 0
        count_0 = len(df[(df["product_type"] == product_type) & (df["county"] == county) & (df["is_business"] == 0)])
        matrix_business_0[j, i] = count_0

        # Count for is_business = 1
        count_1 = len(df[(df["product_type"] == product_type) & (df["county"] == county) & (df["is_business"] == 1)])
        matrix_business_1[j, i] = count_1

# Create DataFrames from the matrices
columns = [f"Product_Type_{i}" for i in product_types]
df_heatmap_0 = pd.DataFrame(matrix_business_0, index=counties, columns=columns)
df_heatmap_1 = pd.DataFrame(matrix_business_1, index=counties, columns=columns)

# Plot heatmaps side by side
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# Plot for is_business = 0
sns.heatmap(df_heatmap_0, annot=True, cmap='Dark2', fmt='.1f', linewidths=.5, ax=axs[0])
axs[0].set_title('Heatmap for is_business = 0')

# Plot for is_business = 1
sns.heatmap(df_heatmap_1, annot=True, cmap='Dark2', fmt='.1f', linewidths=.5, ax=axs[1])
axs[1].set_title('Heatmap for is_business = 1')

plt.show()
'''


##############################################################################################
#                                      weather_station_to_county_mapping
##############################################################################################

'''
# Read the data
data_weather_station_to_county_mapping = pd.read_csv(data_path_weather_station_to_county_mapping)

# Copy the Data
df_weather_station_to_county_mapping = data_weather_station_to_county_mapping.copy()

# Observe few lines 
print(df_weather_station_to_county_mapping)

# Shape of the data
print('The shape of df is:', df_weather_station_to_county_mapping.shape)


# Create columns
Column_name_weather_station_to_county_mapping = list(df_weather_station_to_county_mapping.columns)

# Number of NaN in each column
number_na_weather_station_to_county_mapping = df_weather_station_to_county_mapping.isna().sum()

# Type of Data and the number
types = df_weather_station_to_county_mapping.dtypes
number_types = df_weather_station_to_county_mapping.dtypes.value_counts()
print(number_types)

# Create a resume table
df_resume = pd.DataFrame({'features': Column_name_weather_station_to_county_mapping, 'Type': types, 'Number of NaN': number_na_weather_station_to_county_mapping })
print(df_resume)


df_weather_station_to_county_mapping = df_weather_station_to_county_mapping.dropna(axis=0)

# Observe 
print(df_weather_station_to_county_mapping.sort_values(['county']))

# count 
print(df_weather_station_to_county_mapping.value_counts(['county']).sort_values())

# Shape of the data
print('The shape of df is:', df_weather_station_to_county_mapping.shape)
'''



