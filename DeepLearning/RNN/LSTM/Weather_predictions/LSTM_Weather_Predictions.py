import tensorflow as tf
import keras

from keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math, random


import h5py, json
import os,time,sys


from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split


from datetime import datetime




##############################################################################################
#                                          READ DATA 
##############################################################################################


data_dir = '/Users/jordanmoles/Documents/GitHub/Artificial_Intelligence/DeepLearning/RNN/LSTM/Weather_Predictions/'
csv_path = '/Users/jordanmoles/Documents/GitHub/Artificial_Intelligence/DeepLearning/RNN/LSTM/Weather_Predictions/essential_data_synop.csv'
run_path = '/Users/jordanmoles/Documents/GitHub/Artificial_Intelligence/DeepLearning/RNN/LSTM/Weather_Predictions/models/best-model.h5'


df = pd.read_csv(csv_path, header = 0, sep=';')


# Visualization
#print(df.shape)
#print(df.head())


# Create columns
columns_name= list(df.columns)
#print(columns_name)


# Count number of NA per columns
columns_na = df.isna().sum().tolist()
#print(columns_na)


# Create a table
df_resume = pd.DataFrame({'Features': columns_name, 'Number of Na' : columns_na})
#print(df_resume)





##############################################################################################
#                                       DATA PREPROCESSING
##############################################################################################



########## Removing useless columns ##########


# Suppose 'non_numeric_column' is the non-numeric column in the DataFrame
non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns

# Exclude the 'Date' column from encoding
non_numeric_columns = non_numeric_columns.drop('Date', errors='ignore')

# Encode and fit for each non-numeric column
label_encoder = LabelEncoder()
df_encoded = df.copy()  # Create a copy of the DataFrame to avoid modifying the original
for col in non_numeric_columns:
    df_encoded[col] = label_encoder.fit_transform(df[col])

# Removing useless columns (excluding 'Date')
Remover = VarianceThreshold(threshold=0)
df_transformed = Remover.fit_transform(df_encoded.drop('Date', axis=1))

# Display the transformed DataFrame
df_transformed = pd.DataFrame(df_transformed, columns=np.array(df_encoded.columns[:-1])[Remover.get_support()])
df_transformed['Date'] = df['Date']
print(df_transformed)


########## Clean the Dataset ##########


# Sort the date
df_transformed = df_transformed.sort_values(['Date'])
df_transformed = df_transformed.reset_index(drop=True)


# Interpolate NaN values
df_transformed = df_transformed.interpolate()

# Display the DataFrame
#print(df_transformed)

# Create columns
columns_name= list(df_transformed.columns)
#print(columns_name)


# Count number of NA per columns
columns_na = df_transformed.isna().sum().tolist()
#print(columns_na)


# Create a table
df_transformed_resume = pd.DataFrame({'Features': columns_name, 'Number of Na' : columns_na})
#print(df_transformed_resume)






########## Remove again some columns and some rows where we cannot interpolate ##########

# Drop some columns and rows
df_transformed = df_transformed.drop(['Géopotentiel', 'Hauteur de base 2','Hauteur de base 3', 'Nébulosité couche nuageuse 3',
                                    'Nébulosité couche nuageuse 4', 'Type nuage 3','Type nuage 4', 'Température du thermomètre mouillé',
                                     'region (code)'], axis = 1)
df_transformed.dropna(axis = 0, inplace= True)


# Create columns
columns_name= list(df_transformed.columns)
#print(columns_name)


# Count number of NA per columns
columns_na = df_transformed.isna().sum().tolist()
#print(columns_na)


# Create a table and observe
df_transformed_resume = pd.DataFrame({'Features': columns_name, 'Number of Na' : columns_na})
#print(df_transformed_resume)


# Get the localisation of the columns Pressure, Humidity and Temperature

column_temp = df_transformed.columns.get_loc('Température (°C)')
column_press = df_transformed.columns.get_loc('Pression au niveau mer')
column_hum = df_transformed.columns.get_loc('Humidité')

print('Numéro de colonne température:', column_temp)
print('Numéro de colonne pression:', column_press)
print('Numéro de colonne humidité:', column_hum)




##############################################################################################
#                                        VISUALIZATION
##############################################################################################


# Be sure that 'Date' has the DateTime format
df_transformed['Date']=pd.to_datetime(df_transformed['Date'])


# Filter the data for a particular month (for example, the first one)
start_date = df_transformed['Date'].min()
end_date = start_date + pd.DateOffset(months=1)
df_filtered = df_transformed[(df_transformed['Date'] >= start_date) & (df_transformed['Date'] <= end_date)]


# Display the plot for temperature, pressure, humidity during one month
plt.figure(figsize=(10, 8))

# Subplot for Temperature
plt.subplot(3, 1, 1)  # 3 rows, 1 column, 1st subplot
plt.plot(df_filtered['Date'], df_filtered['Température (°C)'], label='Temperature (°C)', c='blue')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.title('Temperature over time for one month')
plt.legend()
plt.tight_layout()  # Adjust layout for better spacing
plt.grid(ls='--')

# Subplot for Pressure
plt.subplot(3, 1, 2)  # 3 rows, 1 column, 2nd subplot
plt.plot(df_filtered['Date'], df_filtered['Pression au niveau mer'], label='Pression au niveau mer', c='r')
plt.xlabel('Date')
plt.ylabel('Pression au niveau mer')
plt.title('Pressure over time for one month')
plt.legend()
plt.tight_layout()  # Adjust layout for better spacing
plt.grid(ls='--')

# Subplot for Humidity
plt.subplot(3, 1, 3)  # 3 rows, 1 column, 3rd subplot
plt.plot(df_filtered['Date'], df_filtered['Humidité'], label='Humidity', c='green')
plt.xlabel('Date')
plt.ylabel('Humidity')
plt.title('Humidity over time for one month')
plt.legend()

plt.tight_layout()  # Adjust layout for better spacing
plt.grid(ls='--')
#plt.show()







##############################################################################################
#                                           PARAMETERS
##############################################################################################




scale = 1 
train_prop = 0.8
sequence_len = 16
batch_size = 32
epochs = 10
fit_verbosity = 1

features = list(df_transformed.columns.drop(['Date']))
features_len = len(features)








##############################################################################################
#                                          BEFORE THE MODEL
##############################################################################################


# creation of the Train/Test set 
train_len = int(train_prop*df_transformed.shape[0]) 
df_train = df_transformed.loc[:train_len - 1, features]
df_test = df_transformed.loc[train_len:, features]


# We use standardization
# Scale numeric columns only
Scaler = StandardScaler()
df_train_scl = Scaler.fit_transform(df_train)
df_test_scl = Scaler.transform(df_test)


print('Train DataSet:', df_train_scl.shape)
print('Test DataSet:', df_test_scl.shape)



# And create Time data generator
train_generator = TimeseriesGenerator(df_train_scl, df_train_scl, length = sequence_len, batch_size=batch_size)
test_generator = TimeseriesGenerator(df_test_scl, df_test_scl, length = sequence_len, batch_size=batch_size)




##############################################################################################
#                      CREATE,INTRODUCE CALLBACKS, COMPILE AND TRAIN THE RNN 
##############################################################################################
'''

# We will take a series of 16 vectors, each consisting of 47 components (without the Date because it's not interesting for the learning phase), and predict the 17th.
model = keras.models.Sequential([ 
    keras.layers.InputLayer(input_shape=(sequence_len, features_len)),
    keras.layers.LSTM(100, activation='relu'),
    keras.layers.Dropout(0.2), 
    keras.layers.Dense(features_len)
    ])


# Let's observe the parameters
model.summary()



####################### Create the callbacks 
# Create directories
os.makedirs(os.path.join(data_dir, 'models'), exist_ok=True)
os.makedirs(os.path.join(data_dir, 'logs'), exist_ok=True)

# TensorBoard Callback
log_dir = os.path.join(data_dir, "logs", "tb_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# ModelCheckpoint Callback - Save the best model based on validation metric
bestmodel_checkpoint_dir = os.path.join(data_dir, "models", "best-model.h5")
bestmodel_callback = tf.keras.callbacks.ModelCheckpoint(filepath=bestmodel_checkpoint_dir,
                                                         verbose=0,
                                                         monitor='val_mae',  # Use the validation metric
                                                         save_best_only=True)

# ModelCheckpoint Callback - Save the model at each epoch
checkpoint_dir = os.path.join(data_dir, "models", "model-{epoch:04d}.h5")
savemodel_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir, verbose=0)

# Display the command to run TensorBoard
tensorboard_command = f'tensorboard --logdir {os.path.abspath(log_dir)}'
print(f'To run TensorBoard, use the following command:\n{tensorboard_command}')



# Definition of a model-saving function
def save_model(model):
    model.save('my_model.h5')
    print("Model saved successfully.")



# Compile the model
model.compile(optimizer = 'adam',
              loss = 'mse',
              metrics = ['mae'])


# Train the model
history = model.fit(train_generator,
                    batch_size=batch_size,
                    epochs = epochs,
                    verbose =0,
                    validation_data = test_generator,
                    callbacks=[tensorboard_callback,bestmodel_callback,savemodel_callback])

print(history)



score = model.evaluate(test_generator, verbose = 0)


print('Test loss: {:5.4f}'.format(score[0]))
print('Test mae: {:5.4f}'.format(score[1]))


# Save the model
save_model(model)



##############################################################################################
#                                       VISUALIZATION 
##############################################################################################


# Retrieving training and validation metrics
train_mae = history.history['mae']
train_loss = history.history['loss']
val_mae = history.history['val_mae']
val_loss = history.history['val_loss']

# Visualizing the evolution of mae
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_mae, label='Train mae')
plt.plot(val_mae, label='Validation mae')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Visualizing the evolution of loss
plt.subplot(1, 2, 2)
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
'''



##############################################################################################
#                                      MAKE A PREDICTION 
##############################################################################################

loaded_model = tf.keras.models.load_model(run_path)

# Get a sequence
s=random.randint(0,len(df_test)-sequence_len)

sequence      = df_test_scl[s:s+sequence_len]
sequence_true = df_test_scl[s:s+sequence_len+1]

# Prediction
pred = loaded_model.predict( np.array([sequence]) )

print(pred[:,45])

# Plot the actual values and the predicted value
plt.figure(figsize=(8, 6))
plt.plot(range(sequence_len+1), sequence_true[:, 45], label='Actual Values', marker='o')
plt.plot(sequence_len+1, pred[:,45], label='Predicted Value', marker='x', color='red')
plt.xlabel('Time Step')
plt.ylabel('Temperature (°C)')
plt.title('Actual vs Predicted Temperature for a Sequence')
plt.grid('--')
plt.legend()
plt.show()
