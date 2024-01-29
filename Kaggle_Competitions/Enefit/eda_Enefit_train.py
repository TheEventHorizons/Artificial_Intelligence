import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import polars as pl
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







############################################################################################################################################################################################
#                                                                                       FORM ANALYSIS
############################################################################################################################################################################################




##############################################################################################
#                                      Train
##############################################################################################



# Read the data
data_train = pd.read_csv(data_path_train, parse_dates=['datetime'], index_col='datetime')
# Copy the Data
df = data_train.copy()





##############################################################################################
#                                      weather_station_to_county_mapping
##############################################################################################




# Read the data
data_weather_station_to_county_mapping = pd.read_csv(data_path_weather_station_to_county_mapping)

# Copy the Data
df_weather_station_to_county_mapping = data_weather_station_to_county_mapping.copy()


df_weather_station_to_county_mapping['latitude'] = df_weather_station_to_county_mapping['latitude'].round(1)

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


