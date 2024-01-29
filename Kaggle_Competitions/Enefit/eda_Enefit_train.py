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

root = '/Users/jordanmoles/Documents/Programmes_Informatiques/Python/Projects/Kaggle_Competitions/predict-energy-behavior-of-prosumers/'



# Load Dataframes
df_train = pd.read_csv(os.path.join(root, "train.csv")).copy()
df_gas_prices = pd.read_csv(os.path.join(root, "gas_prices.csv")).copy()
df_client = pd.read_csv(os.path.join(root, "client.csv")).copy()
df_electricity_prices = pd.read_csv(os.path.join(root, "electricity_prices.csv")).copy()
df_forecast_weather = pd.read_csv(os.path.join(root, "forecast_weather.csv")).copy()
df_historical_weather = pd.read_csv(os.path.join(root, "historical_weather.csv")).copy()
df_weather_station_to_county_mapping = pd.read_csv(data_path_weather_station_to_county_mapping).copy()



df_weather_station_to_county_mapping['latitude'] = df_weather_station_to_county_mapping['latitude'].round(1)
df_weather_station_to_county_mapping = df_weather_station_to_county_mapping.dropna(axis=0)


# Associate latitude and longitude to counties
df_historical_weather=pd.merge(df_historical_weather, df_weather_station_to_county_mapping[['latitude', 'longitude', 'county']], on=['latitude', 'longitude'], how='left')
print(df_historical_weather)









'''
# Group by multiple columns and aggregate mean for each dataframe
grouped_train = df_train.groupby(['county', 'is_business', 'product_type', 'is_consumption', 'datetime', 'data_block_id', 'prediction_unit_id']).agg({'target': 'mean'}).reset_index()
grouped_gas_price = df_gas_prices.groupby(['forecast_date', 'origin_date', 'data_block_id']).agg({'lowest_price_per_mwh': 'mean', 'highest_price_per_mwh':'mean'}).reset_index()
grouped_client = df_client.groupby(['product_type', 'county', 'is_business', 'date', 'data_block_id']).agg({'eic_count':'mean', 'installed_capacity':'mean'}).reset_index()
grouped_electricity_prices = df_electricity_prices.groupby(['forecast_date', 'origin_date', 'data_block_id']).agg({'euros_per_mwh':'mean'}).reset_index()
grouped_forecast_weather = df_forecast_weather.groupby(['latitude', 'longitude', 'origin_datetime', 'forecast_datetime', 'data_block_id']).agg({'temperature': 'mean','dewpoint': 'mean','cloudcover_high': 'mean','cloudcover_low': 'mean','cloudcover_mid': 'mean','cloudcover_total': 'mean','10_metre_u_wind_component': 'mean','10_metre_v_wind_component': 'mean','direct_solar_radiation': 'mean','surface_solar_radiation_downwards': 'mean','snowfall': 'mean','total_precipitation': 'mean'}).reset_index()
grouped_historical_weather = df_historical_weather.groupby(['latitude', 'longitude', 'datetime', 'data_block_id']).agg({'rain':'mean', 'surface_pressure': 'mean','cloudcover_total': 'mean','cloudcover_high': 'mean','cloudcover_low': 'mean','cloudcover_mid': 'mean','windspeed_10m': 'mean','winddirection_10m': 'mean','shortwave_radiation': 'mean','direct_solar_radiation': 'mean','diffuse_radiation': 'mean','snowfall': 'mean'}).reset_index()
'''