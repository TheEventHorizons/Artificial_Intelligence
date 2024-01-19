import os
import csv
import math, random
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import time
import json
from tensorflow.keras.optimizers import Adam
from pathlib import Path
import pathlib
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from skimage.filters import rank
from skimage import io, color, exposure, transform
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import keras
from keras.callbacks import TensorBoard
import keras.preprocessing.image 
import cv2




from keras.preprocessing.image import ImageDataGenerator




data_dir = 'your/folder/path/archive/reduced/'
data_dir_train = 'your/folder/path/archive/reduced/train'
data_dir_val = 'your/folder/path/archive/reduced/validation'


emotions = ['sad','happy','anger','neutral']

num_class = 4






#####################################################################
# Parameters
#####################################################################



batch_size = 40

epochs = 10

img_size = (48, 48)

scale = 0.1

fit_verbosity = 0

with_datagen  = False

tag_id = 'last'




#####################################################################
# DATA AUGMENTATION
#####################################################################



# Create an image generator for training with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,          # Normalize pixel values between 0 and 1
    shear_range=0.2,         # Shear effect
    zoom_range=0.2,          # Zoom effect
    horizontal_flip=True,    # Horizontal flip
    
)

# Create an image generator for validation without data augmentation
val_datagen = ImageDataGenerator(rescale=1./255)

# Load training and validation images
train_generator = train_datagen.flow_from_directory(
    data_dir_train,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'  # Use 'categorical' for categorical labels
)

validation_generator = val_datagen.flow_from_directory(
    data_dir_val,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)








#####################################################################
# CALLBACKS
#####################################################################


# It's possible to save the model each epoch or at each improvement. The model can be saved completely or partially. 
# For full format we can use HDF5 format


from datetime import datetime

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
                                                         monitor='val_accuracy',  # Use the validation metric
                                                         save_best_only=True)

# ModelCheckpoint Callback - Save the model at each epoch
checkpoint_dir = os.path.join(data_dir, "models", "model-{epoch:04d}.h5")
savemodel_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir, verbose=0)

# Display the command to run TensorBoard
tensorboard_command = f'tensorboard --logdir {os.path.abspath(log_dir)}'
print(f'To run TensorBoard, use the following command:\n{tensorboard_command}')









#####################################################################
# PREPROCESSING
#####################################################################





train_data = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir_train,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=img_size,
    batch_size=batch_size,
    shuffle=True
)



val_data = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir_val,
    validation_split = 0.2,
    subset = "validation",
    seed = 42,
    image_size = img_size,
    batch_size = batch_size,
)



class_names = train_data.class_names
print(class_names)

# Get a batch of images and labels from the train_data
for images, labels in train_data.take(1):
    # Display the first 4 images
    plt.figure(figsize=(10, 10))
    for i in range(4):
        ax = plt.subplot(1, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])  # Convert categorical labels to index
        plt.axis("off")

plt.show()









#####################################################################
# The models
#####################################################################



# Model 1




model_v1 = tf.keras.Sequential([
    keras.layers.Lambda(lambda x: tf.image.rgb_to_grayscale(x), input_shape=(img_size[0], img_size[1], 3)),
    keras.layers.experimental.preprocessing.Rescaling(1./255),
    keras.layers.Conv2D(128, 4, activation='relu'),
    keras.layers.MaxPool2D(),
    keras.layers.Dropout(0.2),

    keras.layers.Conv2D(64, 4, activation='relu'),
    keras.layers.MaxPool2D(),
    keras.layers.Dropout(0.2),

    keras.layers.Conv2D(32, 4, activation='relu'),
    keras.layers.MaxPool2D(),
    keras.layers.Dropout(0.2),


    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),

    keras.layers.Dense(num_class,activation='softmax')


])



#model_v1.compile(optimizer='adam',
#              loss='sparse_categorical_crossentropy',
#              metrics = ['accuracy'])


#history = model_v1.fit(train_data,
#                    validation_data = val_data,
#                    epochs = epochs,
#                    batch_size = batch_size,
#                    verbose = 1,
#                    callbacks=[tensorboard_callback,bestmodel_callback,savemodel_callback])

#print(history)


#model_v1.summary()


#score = model_v1.evaluate(val_data, verbose = 0)


#print('Test loss: {:5.4f}'.format(score[0]))
#print('Test accuracy: {:5.4f}'.format(score[1]))






# Model 2





model_v2 = tf.keras.Sequential([
    keras.layers.Lambda(lambda x: tf.image.rgb_to_grayscale(x), input_shape=(img_size[0], img_size[1], 3)),
    keras.layers.experimental.preprocessing.Rescaling(1./255),
    keras.layers.Conv2D(64, 4, activation='relu'),
    keras.layers.MaxPool2D(),
    keras.layers.Dropout(0.2),

    keras.layers.Conv2D(128, 4, activation='relu'),
    keras.layers.MaxPool2D(),
    keras.layers.Dropout(0.2),

    keras.layers.Conv2D(256, 4, activation='relu'),
    keras.layers.MaxPool2D(),
    keras.layers.Dropout(0.2),


    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.5),

    keras.layers.Dense(num_class,activation='softmax')


])



model_v2.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics = ['accuracy'])


history = model_v2.fit(train_data,
                    validation_data = val_data,
                    epochs = epochs,
                    batch_size = batch_size,
                    verbose = 1,
                    callbacks=[tensorboard_callback,bestmodel_callback,savemodel_callback])

print(history)


model_v2.summary()


score = model_v2.evaluate(val_data, verbose = 0)


print('Test loss: {:5.4f}'.format(score[0]))
print('Test accuracy: {:5.4f}'.format(score[1]))






# Model 3







model_v3 = tf.keras.Sequential([
    keras.layers.Lambda(lambda x: tf.image.rgb_to_grayscale(x), input_shape=(img_size[0], img_size[1], 3)),
    keras.layers.experimental.preprocessing.Rescaling(1./255),
    keras.layers.Conv2D(512, 4, activation='relu'),
    keras.layers.MaxPool2D(),
    keras.layers.Dropout(0.2),

    keras.layers.Conv2D(256, 4, activation='relu'),
    keras.layers.MaxPool2D(),
    keras.layers.Dropout(0.2),

    keras.layers.Conv2D(128, 4, activation='relu'),
    keras.layers.MaxPool2D(),
    keras.layers.Dropout(0.2),


    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),

    keras.layers.Dense(num_class,activation='softmax')


])



#model_v3.compile(optimizer='adam',
#              loss='sparse_categorical_crossentropy',
#              metrics = ['accuracy'])


#history = model_v3.fit(train_data,
#                    validation_data = val_data,
#                    epochs = epochs,
#                    batch_size = batch_size,
#                    verbose = 1,
#                    callbacks=[tensorboard_callback,bestmodel_callback,savemodel_callback])

#print(history)


#model_v3.summary()


#score = model_v3.evaluate(val_data, verbose = 0)


#print('Test loss: {:5.4f}'.format(score[0]))
#print('Test accuracy: {:5.4f}'.format(score[1]))












model_architectures = {'v1': model_v1, 'v2':model_v2}








#####################################################################
# MULTIPLE MODEL RUN
#####################################################################

'''


# Create an empty DataFrame to store the results
results_df = pd.DataFrame(index=['Loss', 'Accuracy', 'Time'])


# Iterate over models
for model_name, model in model_architectures.items():
    model_dir = os.path.join(data_dir, 'models', model_name)
    os.makedirs(model_dir, exist_ok=True)

    # TensorBoard Callback
    log_dir = os.path.join(data_dir, 'logs', model_name, "tb_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # ModelCheckpoint Callback - Save the best model based on validation metric
    bestmodel_checkpoint_dir = os.path.join(model_dir, "best-model.h5")
    bestmodel_callback = tf.keras.callbacks.ModelCheckpoint(filepath=bestmodel_checkpoint_dir,
                                                             verbose=0,
                                                             monitor='val_accuracy',  # Use the validation metric
                                                             save_best_only=True)

    # ModelCheckpoint Callback - Save the model at each epoch
    checkpoint_dir = os.path.join(model_dir, "model-{epoch:04d}.h5")
    savemodel_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir, verbose=0)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    start_time = time.time()
    history = model.fit(train_data,
                        validation_data=val_data,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=1,
                        callbacks=[tensorboard_callback, bestmodel_callback, savemodel_callback])
    end_time = time.time()

    # Display scores
    score = model.evaluate(val_data, verbose=0)
    loss = score[0]
    accuracy = score[1]
    training_time = end_time - start_time

    # Add results to the DataFrame
    results_df[model_name] = [loss, accuracy, training_time]

    print(f"\n{model_name} Results:")
    print("Loss: {:5.4f}".format(loss))
    print('Accuracy: {:5.4f}'.format(accuracy))
    print("Training Time: {:.2f} seconds".format(training_time))

# Display the results table
print("\nResults Table:")
print(results_df)

'''








#####################################################################
# VISUALIZATION 
#####################################################################



# Retrieving training and validation metrics
train_accuracy = history.history['accuracy']
train_loss = history.history['loss']
val_accuracy = history.history['val_accuracy']
val_loss = history.history['val_loss']

# Visualizing the evolution of accuracy
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_accuracy, label='Train Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
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




#####################################################################
#TEST
#####################################################################


# Load a single image from the validation set
sample_image, sample_label = next(iter(validation_generator))

# Expand dimensions for prediction
sample_image = np.expand_dims(sample_image[0], axis=0)

# Make prediction
predictions = model_v2.predict(sample_image)


# Get the predicted class and probabilities
predicted_class = np.argmax(predictions)
predicted_probabilities = predictions[0]

print(predicted_probabilities)

# Get the actual class
actual_class = np.argmax(sample_label[0])

# Visualize the results
plt.imshow(np.squeeze(sample_image))
plt.title(f'Predicted Class: {class_names[predicted_class]}, Actual Class: {class_names[actual_class]}')
plt.axis('off')
plt.show()
