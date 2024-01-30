import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import polars as pl
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

data_cols = [
        "target",
        "county",
        "is_business",
        "product_type",
        "is_consumption",
        "datetime",
        "row_id",
        ]
client_cols = [
        "product_type",
        "county",
        "eic_count",
        "installed_capacity",
        "is_business",
        "date",
        ]
gas_prices_cols = ["forecast_date", "lowest_price_per_mwh", "highest_price_per_mwh"]
electricity_prices_cols = ["forecast_date", "euros_per_mwh"]
forecast_weather_cols = [
        "latitude",
        "longitude",
        "hours_ahead",
        "temperature",
        "dewpoint",
        "cloudcover_high",
        "cloudcover_low",
        "cloudcover_mid",
        "cloudcover_total",
        "10_metre_u_wind_component",
        "10_metre_v_wind_component",
        "forecast_datetime",
        "direct_solar_radiation",
        "surface_solar_radiation_downwards",
        "snowfall",
        "total_precipitation",
        ]
historical_weather_cols = [
        "datetime",
        "temperature",
        "dewpoint",
        "rain",
        "snowfall",
        "surface_pressure",
        "cloudcover_total",
        "cloudcover_low",
        "cloudcover_mid",
        "cloudcover_high",
        "windspeed_10m",
        "winddirection_10m",
        "shortwave_radiation",
        "direct_solar_radiation",
        "diffuse_radiation",
        "latitude",
        "longitude",
        ]
location_cols = ["longitude", "latitude", "county"]
target_cols = [
        "target",
        "county",
        "is_business",
        "product_type",
        "is_consumption",
        "datetime",
        ]



df_data = pl.read_csv(os.path.join(root, "train.csv"), columns=data_cols, try_parse_dates=True)
df_client = pl.read_csv(os.path.join(root, "client.csv"), columns=client_cols, try_parse_dates=True)
df_gas_prices = pl.read_csv(os.path.join(root, "gas_prices.csv"), columns=gas_prices_cols, try_parse_dates=True)
df_electricity_prices = pl.read_csv(os.path.join(root, "electricity_prices.csv"), columns=electricity_prices_cols, try_parse_dates=True)
df_forecast_weather = pl.read_csv(os.path.join(root, "forecast_weather.csv"), columns=forecast_weather_cols, try_parse_dates=True)
df_historical_weather = pl.read_csv(os.path.join(root, "historical_weather.csv"), columns=historical_weather_cols, try_parse_dates=True)
df_weather_station_to_county_mapping = pl.read_csv(os.path.join(root, "weather_station_to_county_mapping.csv"), columns=location_cols, try_parse_dates=True)
df_target = df_data.select(target_cols)

schema_data = df_data.schema
schema_client = df_client.schema
schema_gas  = df_gas_prices.schema
schema_electricity = df_electricity_prices.schema
schema_forecast = df_forecast_weather.schema
schema_historical = df_historical_weather.schema
schema_target = df_target.schema





def generate_features(
        df_data, 
        df_client, 
        df_gas_prices, 
        df_electricity_prices, 
        df_forecast_weather, 
        df_historical_weather, 
        df_weather_station_to_county_mapping, 
        df_target
):
    df_data = (
        df_data
        .with_columns(
            pl.col("datetime").cast(pl.Date).alias("date"),
        )
    )
    
    df_gas_prices = (
        df_gas_prices
        .rename({"forecast_date": "date"})
    )
    
    df_electricity_prices = (
        df_electricity_prices
        .rename({"forecast_date": "datetime"})
    )
    
    df_weather_station_to_county_mapping = (
        df_weather_station_to_county_mapping
        .with_columns(
            pl.col("latitude").cast(pl.datatypes.Float32),
            pl.col("longitude").cast(pl.datatypes.Float32)
        )
    )
    
    # sum of all product_type targets related to ["datetime", "county", "is_business", "is_consumption"]
    df_target_all_type_sum = (
        df_target
        .group_by(["datetime", "county", "is_business", "is_consumption"]).sum()
        .drop("product_type")
    )
    
    df_forecast_weather = (
        df_forecast_weather
        .rename({"forecast_datetime": "datetime"})
        .filter(pl.col("hours_ahead") >= 24) # we don't need forecast for today
        .with_columns(
            pl.col("latitude").cast(pl.datatypes.Float32),
            pl.col("longitude").cast(pl.datatypes.Float32),
            # datetime for forecast in a different timezone
            pl.col('datetime').dt.replace_time_zone(None).cast(pl.Datetime("us"))
        )
        .join(df_weather_station_to_county_mapping, how="left", on=["longitude", "latitude"])
        .drop("longitude", "latitude")
    )
    
    df_historical_weather = (
        df_historical_weather
        .with_columns(
            pl.col("latitude").cast(pl.datatypes.Float32),
            pl.col("longitude").cast(pl.datatypes.Float32),
        )
        .join(df_weather_station_to_county_mapping, how="left", on=["longitude", "latitude"])
        .drop("longitude", "latitude")
    )
    
    # creating average forecast characteristics for all weather stations
    df_forecast_weather_date = (
        df_forecast_weather
        .group_by("datetime").mean()
        .drop("county")
    )
    
    # creating average forecast characteristics for weather stations related to county
    df_forecast_weather_local = (
        df_forecast_weather
        .filter(pl.col("county").is_not_null())
        .group_by("county", "datetime").mean()
    )
    
    # creating average historical characteristics for all weather stations
    df_historical_weather_date = (
        df_historical_weather
        .group_by("datetime").mean()
        .drop("county")
    )
    
    # creating average historical characteristics for weather stations related to county
    df_historical_weather_local = (
        df_historical_weather
        .filter(pl.col("county").is_not_null())
        .group_by("county", "datetime").mean()
    )
    
    df_data = (
        df_data
        # pl.duration(days=1) shifts datetime to join lag features (usually we join last available values)
        .join(df_gas_prices.with_columns((pl.col("date") + pl.duration(days=1)).cast(pl.Date)), on="date", how="left")
        .join(df_client.with_columns((pl.col("date") + pl.duration(days=2)).cast(pl.Date)), on=["county", "is_business", "product_type", "date"], how="left")
        .join(df_electricity_prices.with_columns(pl.col("datetime") + pl.duration(days=1)), on="datetime", how="left")
        
        # lag forecast_weather features (24 hours * days)
        .join(df_forecast_weather_date, on="datetime", how="left", suffix="_fd")
        .join(df_forecast_weather_local, on=["county", "datetime"], how="left", suffix="_fl")
        .join(df_forecast_weather_date.with_columns(pl.col("datetime") + pl.duration(days=7)), on="datetime", how="left", suffix="_fd_7d")
        .join(df_forecast_weather_local.with_columns(pl.col("datetime") + pl.duration(days=7)), on=["county", "datetime"], how="left", suffix="_fl_7d")

        # lag historical_weather features (24 hours * days)
        .join(df_historical_weather_date.with_columns(pl.col("datetime") + pl.duration(days=2)), on="datetime", how="left", suffix="_hd_2d")
        .join(df_historical_weather_local.with_columns(pl.col("datetime") + pl.duration(days=2)), on=["county", "datetime"], how="left", suffix="_hl_2d")
        .join(df_historical_weather_date.with_columns(pl.col("datetime") + pl.duration(days=7)), on="datetime", how="left", suffix="_hd_7d")
        .join(df_historical_weather_local.with_columns(pl.col("datetime") + pl.duration(days=7)), on=["county", "datetime"], how="left", suffix="_hl_7d")
        
        # lag target features (24 hours * days)
        .join(df_target.with_columns(pl.col("datetime") + pl.duration(days=2)).rename({"target": "target_1"}), on=["county", "is_business", "product_type", "is_consumption", "datetime"], how="left")
        .join(df_target.with_columns(pl.col("datetime") + pl.duration(days=3)).rename({"target": "target_2"}), on=["county", "is_business", "product_type", "is_consumption", "datetime"], how="left")
        .join(df_target.with_columns(pl.col("datetime") + pl.duration(days=4)).rename({"target": "target_3"}), on=["county", "is_business", "product_type", "is_consumption", "datetime"], how="left")
        .join(df_target.with_columns(pl.col("datetime") + pl.duration(days=5)).rename({"target": "target_4"}), on=["county", "is_business", "product_type", "is_consumption", "datetime"], how="left")
        .join(df_target.with_columns(pl.col("datetime") + pl.duration(days=6)).rename({"target": "target_5"}), on=["county", "is_business", "product_type", "is_consumption", "datetime"], how="left")
        .join(df_target.with_columns(pl.col("datetime") + pl.duration(days=7)).rename({"target": "target_6"}), on=["county", "is_business", "product_type", "is_consumption", "datetime"], how="left")
        .join(df_target.with_columns(pl.col("datetime") + pl.duration(days=14)).rename({"target": "target_7"}), on=["county", "is_business", "product_type", "is_consumption", "datetime"], how="left")
        
        .join(df_target_all_type_sum.with_columns(pl.col("datetime") + pl.duration(days=2)).rename({"target": "target_1"}), on=["county", "is_business", "is_consumption", "datetime"], suffix="_all_type_sum", how="left")
        .join(df_target_all_type_sum.with_columns(pl.col("datetime") + pl.duration(days=3)).rename({"target": "target_2"}), on=["county", "is_business", "is_consumption", "datetime"], suffix="_all_type_sum", how="left")
        .join(df_target_all_type_sum.with_columns(pl.col("datetime") + pl.duration(days=7)).rename({"target": "target_6"}), on=["county", "is_business", "is_consumption", "datetime"], suffix="_all_type_sum", how="left")
        .join(df_target_all_type_sum.with_columns(pl.col("datetime") + pl.duration(days=14)).rename({"target": "target_7"}), on=["county", "is_business", "is_consumption", "datetime"], suffix="_all_type_sum", how="left")
        
        
        .with_columns(
            pl.col("datetime").dt.ordinal_day().alias("dayofyear"),
            pl.col("datetime").dt.hour().alias("hour"),
            pl.col("datetime").dt.day().alias("day"),
            pl.col("datetime").dt.weekday().alias("weekday"),
            pl.col("datetime").dt.month().alias("month"),
            pl.col("datetime").dt.year().alias("year"),
        )
        
        .with_columns(
            pl.concat_str("county", "is_business", "product_type", "is_consumption", separator="_").alias("segment"),
        )
        

        .with_columns(
            (np.pi * pl.col("dayofyear") / 183).sin().alias("sin(dayofyear)"),
            (np.pi * pl.col("dayofyear") / 183).cos().alias("cos(dayofyear)"),
            (np.pi * pl.col("hour") / 12).sin().alias("sin(hour)"),
            (np.pi * pl.col("hour") / 12).cos().alias("cos(hour)"),
        )
        
        .with_columns(
            pl.col(pl.Float64).cast(pl.Float32),
        )
        
        .drop("date", "datetime", "hour", "dayofyear")
    )
    
    return df_data


print(generate_features(
        df_data, 
        df_client, 
        df_gas_prices, 
        df_electricity_prices, 
        df_forecast_weather, 
        df_historical_weather, 
        df_weather_station_to_county_mapping, 
        df_target
).columns)










'''

df_weather_station_to_county_mapping['latitude'] = df_weather_station_to_county_mapping['latitude'].round(1)
df_weather_station_to_county_mapping = df_weather_station_to_county_mapping.dropna(axis=0)


# Associate latitude and longitude to counties
df_historical_weather = pd.merge(df_historical_weather, df_weather_station_to_county_mapping[['latitude', 'longitude', 'county']], on=['latitude', 'longitude'], how='left')
df_historical_weather = df_historical_weather.drop(['latitude', 'longitude'], axis = 1)
df_historical_weather = df_historical_weather.dropna(axis=0).reset_index()
df_forecast_weather = pd.merge(df_forecast_weather, df_weather_station_to_county_mapping[['latitude', 'longitude', 'county']], on=['latitude', 'longitude'], how='left')
df_forecast_weather = df_forecast_weather.drop(['latitude', 'longitude'], axis = 1)
df_forecast_weather = df_forecast_weather.dropna(axis=0).reset_index()


# Group by multiple columns and aggregate mean for each dataframe
mean_train = df_train.groupby(['county', 'is_business', 'product_type', 'is_consumption', 'datetime', 'data_block_id', 'prediction_unit_id']).agg({'target': 'mean'}).reset_index()
mean_historical_weather = df_historical_weather.groupby(['county', 'datetime', 'data_block_id']).agg(
    {'rain':'mean', 'surface_pressure': 'mean','cloudcover_total': 'mean','cloudcover_high': 'mean','cloudcover_low': 'mean','cloudcover_mid': 'mean','windspeed_10m': 'mean',
     'winddirection_10m': 'mean','shortwave_radiation': 'mean','direct_solar_radiation': 'mean','diffuse_radiation': 'mean','snowfall': 'mean'}).reset_index()
mean_forecast_weather = df_forecast_weather.groupby(['county', 'origin_datetime', 'forecast_datetime', 'data_block_id']).agg(
    {'temperature': 'mean','dewpoint': 'mean','cloudcover_high': 'mean','cloudcover_low': 'mean','cloudcover_mid': 'mean','cloudcover_total': 'mean','10_metre_u_wind_component': 'mean',
     '10_metre_v_wind_component': 'mean','direct_solar_radiation': 'mean','surface_solar_radiation_downwards': 'mean','snowfall': 'mean','total_precipitation': 'mean'}).reset_index()
mean_gas_price = df_gas_prices.groupby(['forecast_date', 'origin_date', 'data_block_id']).agg({'lowest_price_per_mwh': 'mean', 'highest_price_per_mwh':'mean'}).reset_index()
mean_client = df_client.groupby(['product_type', 'county', 'is_business', 'date', 'data_block_id']).agg({'eic_count':'mean', 'installed_capacity':'mean'}).reset_index()
mean_electricity_prices = df_electricity_prices.groupby(['forecast_date', 'origin_date', 'data_block_id']).agg({'euros_per_mwh':'mean'}).reset_index()


print('train:', mean_train)

merged_data = pd.merge(mean_train, mean_gas_price, on='data_block_id', how='inner')


print('historical weather:', mean_historical_weather)
print('client:', mean_client)
print('electricity:', mean_electricity_prices)
print(merged_data)
'''



