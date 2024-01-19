import os, sys, time
import csv
import math, random

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import time

from pathlib import Path
import pathlib
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from skimage.filters import rank
from skimage import io, color, exposure, transform

import tensorflow as tf
import keras
import keras.preprocessing.image
import os
from pathlib import Path
import shutil

data_dir = '/path/folder/archive'
emotions = ['sad', 'anger', 'happy', 'neutral']



# Reduce the training set to 10%
proportion = 0.1



# Create a new directory for the reduced subset
reduced_data_dir = '/path/folder/archive/reduced'
Path(reduced_data_dir).mkdir(parents=True, exist_ok=True)

# Copy only 10% of the data to the new directory
for emotion in os.listdir(data_dir):
    emotion_folder_path = os.path.join(data_dir, emotion)
    
    # Check if the item is a directory
    if os.path.isdir(emotion_folder_path):
        images = [f for f in os.listdir(emotion_folder_path) if f.endswith('.jpg')]
        
        # Calculate the number of images to copy for each emotion
        num = int(len(images) * proportion)
        images = images[:num]
        
        # Create folders for each emotion in the reduced subset
        reduced_emotion_folder = os.path.join(reduced_data_dir, emotion)
        Path(reduced_emotion_folder).mkdir(parents=True, exist_ok=True)
        
        # Copy images to the new directory
        for image in images:
            src_path = os.path.join(emotion_folder_path, image)
            dest_path = os.path.join(reduced_emotion_folder, image)
            shutil.copy(src_path, dest_path)

print("Creation of the reduced training subset completed.")

# Check if the main directory exists
if not os.path.exists(data_dir):
    print(f"The main directory {data_dir} does not exist.")
    exit()

# Create train and validation folders
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'validation')

# Create folders if they do not exist
Path(train_dir).mkdir(parents=True, exist_ok=True)
Path(val_dir).mkdir(parents=True, exist_ok=True)

# Copy images to the appropriate folders
for emotion in emotions:
    emotion_folder_path = os.path.join(data_dir, emotion)
    images = [f for f in os.listdir(emotion_folder_path) if f.endswith('.jpg')]
    num_images_folder = len(images)
    
    # Calculate the number of images for the train set and the validation set
    num_train = int(num_images_folder * 0.8)
    
    # Select images for the train set
    train_images = images[:num_train]
    # Select images for the validation set
    val_images = images[num_train:]
    
    # Create folders for each emotion in the train and validation folders
    for folder in [train_dir, val_dir]:
        emotion_folder = os.path.join(folder, emotion)
        Path(emotion_folder).mkdir(parents=True, exist_ok=True)
    
    # Copy images to the appropriate folders
    for image in train_images:
        src_path = os.path.join(emotion_folder_path, image)
        dest_path = os.path.join(train_dir, emotion, image)
        shutil.copy(src_path, dest_path)
    
    for image in val_images:
        src_path = os.path.join(emotion_folder_path, image)
        dest_path = os.path.join(val_dir, emotion, image)
        shutil.copy(src_path, dest_path)

print("Creation of train and validation folders completed.")








