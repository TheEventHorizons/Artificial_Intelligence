import tensorflow as tf
import keras
from keras.callbacks import TensorBoard
from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import h5py
import os, time, sys
import random

from importlib import reload




enhanced_dir = '/BenchMarks/archive/data'

dataset_name = 'set-24x24-L'
batch_size = 64
epochs = 10
scale = 1


def read_dataset(enhanced_dir, dataset_name):
    '''Reads h5 dataset
    Args:
        filename     : datasets filename
        dataset_name : dataset name, without .h5
    Returns:    x_train,y_train, x_test,y_test data, x_meta,y_meta'''

    # Read Dataset
    filename = f'{enhanced_dir}/{dataset_name}.h5'
    with h5py.File(filename,'r') as f:
        print(list(f.keys()))
        x_train = f['x_train'][:]
        y_train = f['y_train'][:]
        x_test = f['x_test'][:]
        y_test = f['y_test'][:]
        x_meta = f['x_meta'][:]
        y_meta = f['y_meta'][:]                    
    print(x_train.shape, y_train.shape)


    # Shuffle train set
    train_indices = list(range(len(x_train)))
    random.shuffle(train_indices)

    x_train = x_train[train_indices]
    y_train = y_train[train_indices]
    return  x_train,y_train, x_test,y_test, x_meta,y_meta




## Read dataset
    
X_train, y_train, X_test, y_test, X_meta, y_meta = read_dataset(enhanced_dir, dataset_name)

print(X_train.shape)

class_unique = np.unique(y_train)
images_per_row = 8
num_row = len(class_unique) // images_per_row + int(len(class_unique) % images_per_row != 0)
num_column = images_per_row

fig, ax = plt.subplots(num_row, num_column, figsize=(15, 3 * num_row))

for i, Class_label in enumerate(class_unique):
    row = i // num_column
    column = i % num_column
    Class_index = np.where(y_train == Class_label)[0]
    ax[row, column].imshow(X_train[Class_index[0]])
    ax[row, column].set_title(f'Classe {Class_label}')
    ax[row, column].axis('off')

plt.tight_layout()
plt.show()


# Create a model

def model_v1(lx,ly,lz):

    model = keras.models.Sequential()

    model.add(keras.layers.Input(shape=(lx, ly, lz)))
    model.add( keras.layers.Conv2D(96, (3,3), activation='relu') )
    model.add( keras.layers.MaxPool2D((2,2)) )
    model.add( keras.layers.Dropout(0.2) )


    model.add( keras.layers.Conv2D(192, (3,3), activation='relu') )
    model.add( keras.layers.MaxPool2D((2,2)) )
    model.add( keras.layers.Dropout(0.2) )

    model.add( keras.layers.Flatten() )
    model.add( keras.layers.Dense(1000, activation='relu') )
    model.add( keras.layers.Dropout(0.5) )

    model.add( keras.layers.Dense(43, activation='softmax'))

    return model


(n,lx,ly,lz) = X_train.shape


print("Images of the dataset have this following shape:", (lx,ly,lz))


model = model_v1(lx,ly,lz)

model.summary()


 # Shuffle train set
train_indices = list(range(len(X_train)))
random.shuffle(train_indices)

X_test = X_test/X_train.max()

X_train = X_train[train_indices]/X_train.max()
y_train = y_train[train_indices]




model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

history = model.fit(X_train, y_train,
                    batch_size = batch_size,
                    epochs = epochs,
                    verbose = 1,
                    validation_data=(X_test,y_test))

score = model.evaluate(X_test,y_test, verbose = 0)


print('Test loss: {:5.4f}'.format(score[0]))
print('Test accuracy: {:5.4f}'.format(score[1]))

model.save('/BenchMarks/model.h5')

