import os, sys, time
import csv
import math, random

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import time

from skimage.morphology import disk
from skimage.util import img_as_ubyte
from skimage.filters import rank
from skimage import io, color, exposure, transform
from PIL import Image
from importlib import reload


# For tests on the machine:

scale = 0.1
output_dir = 'BenchMarks/archive/data'



# For a full test generation:

#scale = 1
#output_dir = f'{dataset_dir}/GTSRB/enhanced'


# The original dataset is in : <dataset_dir>/GTSRB/origine.
# There are 3 subsets: Train, Test and Meta.
# Each subset has a csv file and a subdir with images.




df = pd.read_csv('BenchMarks/archive/Test.csv', header=0)

print(df.head())

# This function simply prints the progress percentage to the console, overwriting the previous progress information.

def update_progress(name, current, total):
    progress = current / total
    progress_percent = int(progress * 100)
    sys.stdout.write(f'\r{name}: {progress_percent}% complete')
    sys.stdout.flush()


# Read a subset
def read_csv_dataset(csv_file, percentage=0.1):
    '''
    Reads traffic sign data from German Traffic Sign Recognition Benchmark dataset.
    Arguments:
        csv filename : Description file, Example /data/GTSRB/Train.csv
        percentage   : Percentage of data to use (default is 10%)
    Returns:
        X, y         : np array of images, np array of corresponding labels
    '''
    path = os.path.dirname(csv_file)
    name = os.path.basename(csv_file)

    # Read csv file
    df = pd.read_csv(csv_file, header=0)

    # Get filenames and ClassIds
    total_samples = len(df)
    sample_size = int(percentage * total_samples)

    # Randomly select 10% of the data
    sampled_df = df.sample(n=sample_size, random_state=42)

    filenames = sampled_df['Path'].to_list()
    y = sampled_df['ClassId'].to_list()
    X = []

    # Read Images
    for i, filename in enumerate(filenames, start=1):
        image = io.imread(f'{path}/{filename}')
        X.append(image)
        update_progress(name, i, len(filenames))

    # Return
    return np.array(X, dtype=object), np.array(y)



##################################################



# Read DataSet

(X_train, y_train) = read_csv_dataset('BenchMarks/archive/Train.csv')
(X_test, y_test) = read_csv_dataset('BenchMarks/archive/Test.csv')
(X_meta, y_meta) = read_csv_dataset('BenchMarks/archive/Meta.csv')


# Shuffle train set
train_indices = list(range(len(X_train)))
random.shuffle(train_indices)

X_train = X_train[train_indices]
y_train = y_train[train_indices]


# Sort Meta

combined = list(zip(X_meta,y_meta))
combined.sort(key=lambda x:x[1])
X_meta, y_meta = zip(*combined)







##################################################



# Statistics

train_size = []
train_ratio = []
train_lx = []
train_ly = []

test_size = []
test_ratio = []
test_lx = []
test_ly = []

for image in X_train:
    (lx,ly,lz) = image.shape
    train_size.append(lx*ly/1024)
    train_ratio.append(lx/ly)
    train_lx.append(lx)
    train_ly.append(ly)

for image in X_test:
    (lx,ly,lz) = image.shape
    test_size.append(lx*ly/1024)
    test_ratio.append(lx/ly)
    test_lx.append(lx)
    test_ly.append(ly)


print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# Statistics sizes

plt.figure(figsize=(16,6))
plt.hist([train_size, test_size], bins = 100)
plt.gca().set(title = 'Sizes in Kpixel - Train= [{:5.2f}, {:5.2f}]'.format(min(train_size), max(train_size)), ylabel = 'Population', xlim=[0,30])
plt.legend(['Train', 'Test'])
#plt.show()

# Statistics ratio

plt.figure(figsize=(16,6))
plt.hist([train_ratio, test_ratio], bins = 100)
plt.gca().set(title = 'Ratio lx/ly - Train= [{:5.2f}, {:5.2f}]'.format(min(train_ratio), max(train_ratio)), ylabel = 'Population', xlim=[0.8,1.2])
plt.legend(['Train', 'Test'])
#plt.show()



# Statistics lx

plt.figure(figsize=(16,6))
plt.hist([train_lx, test_lx], bins = 100)
plt.gca().set(title = 'Images lx - Train= [{:5.2f}, {:5.2f}]'.format(min(train_lx), max(train_lx)), ylabel = 'Population', xlim=[20,150])
plt.legend(['Train', 'Test'])
#plt.show()

# Statistics ly
plt.figure(figsize=(16,6))
plt.hist([train_ly,test_ly], bins=100)
plt.gca().set(title='Images ly - Train=[{:5.2f}, {:5.2f}]'.format(min(train_ly),max(train_ly)), 
              ylabel='Population', xlim=[20,150])
plt.legend(['Train','Test'])
#plt.show()

# Statistics classId
plt.figure(figsize=(16,6))
plt.hist([y_train,y_test], bins=43)
plt.gca().set(title='ClassesId', ylabel='Population', xlim=[0,43])
plt.legend(['Train','Test'])
#plt.show()





##################################################






# Display one image for each class in multiple columns with about 8 images per row
class_unique = np.unique(y_train)
images_per_row = 8  # You can adjust the number of images per row here
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

path = 'BenchMarks/archive'

# Display one image for each class in multiple columns with about 8 images per row (from Meta data)
num_classes_meta = 43  # Total number of classes in the Meta folder
images_per_row_meta = 8  # You can adjust the number of images per row here
num_row_meta = num_classes_meta // images_per_row_meta + int(num_classes_meta % images_per_row_meta != 0)
num_column_meta = images_per_row_meta

fig_meta, ax_meta = plt.subplots(num_row_meta, num_column_meta, figsize=(15, 3 * num_row_meta))

for i in range(num_classes_meta):
    row_meta = i // num_column_meta
    column_meta = i % num_column_meta
    image_meta = io.imread(f'{path}/Meta/{i}.png')  # Make sure your images are stored as "0.png", "1.png", ..., "42.png"
    ax_meta[row_meta, column_meta].imshow(image_meta)
    ax_meta[row_meta, column_meta].set_title(f'Classe {i}')
    ax_meta[row_meta, column_meta].axis('off')

plt.tight_layout()
plt.show()




#Images must :
#- have the same size to match the size of the network,be normalized.
#- It is possible to work on rgb or monochrome images and to equalize the histograms.


def images_enhancement(images, width=25, height=25, mode='RGB'):
    '''
    Resize and convert images - doesn't change originals.
    input images must be RGBA or RGB.
    Note : all outputs are fixed size numpy array of float64
    args:
        images :         images list
        width,height :   new images size (25,25)
        mode :           RGB | RGB-HE | L | L-HE | L-LHE | L-CLAHE
    return:
        numpy array of enhanced images
    '''
    modes = { 'RGB':3, 'RGB-HE':3, 'L':1, 'L-HE':1, 'L-LHE':1, 'L-CLAHE':1}
    lz=modes[mode]
    
    out=[]
    for img in images:
        
        # ---- if RGBA, convert to RGB
        if img.shape[2]==4:
            img=color.rgba2rgb(img)
            
        # ---- Resize
        img = transform.resize(img, (width,height))

        # ---- RGB / Histogram Equalization
        if mode=='RGB-HE':
            hsv = color.rgb2hsv(img.reshape(width,height,3))
            hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
            img = color.hsv2rgb(hsv)
        
        # ---- Grayscale
        if mode=='L':
            img=color.rgb2gray(img)
            
        # ---- Grayscale / Histogram Equalization
        if mode=='L-HE':
            img=color.rgb2gray(img)
            img=exposure.equalize_hist(img)
            
        # ---- Grayscale / Local Histogram Equalization
        if mode=='L-LHE':        
            img=color.rgb2gray(img)
            img = img_as_ubyte(img)
            img=rank.equalize(img, disk(10))/255.
        
        # ---- Grayscale / Contrast Limited Adaptive Histogram Equalization (CLAHE)
        if mode=='L-CLAHE':
            img=color.rgb2gray(img)
            img=exposure.equalize_adapthist(img)
            
        # ---- Add image in list of list
        out.append(img)
        update_progress('Enhancement: ', len(out), len(images))

    # ---- Reshape images
    #     (-1, width,height,1) for L
    #     (-1, width,height,3) for RGB
    #
    out = np.array(out,dtype='float64')
    out = out.reshape(-1,width,height,lz)
    return out




i = random.randint(0, len(X_train) - 12)
X_samples = X_train[i:i+12]
y_samples = y_train[i:i+12]



datasets  = {}


datasets['RGB']      = images_enhancement(X_samples, width=25, height=25, mode='RGB')
datasets['RGB-HE']   = images_enhancement(X_samples, width=25, height=25, mode='RGB-HE')
datasets['L']        = images_enhancement(X_samples, width=25, height=25, mode='L')
datasets['L-HE']     = images_enhancement(X_samples, width=25, height=25, mode='L-HE')
datasets['L-LHE']    = images_enhancement(X_samples, width=25, height=25, mode='L-LHE')
datasets['L-CLAHE']  = images_enhancement(X_samples, width=25, height=25, mode='L-CLAHE')






# Create a global figure
plt.figure(figsize=(15, 36))

# Display 12 images from the Meta folder
for i in range(12):
    image_meta = io.imread(f'{path}/Meta/{i}.png')
    plt.subplot(3, 12, i + 1)
    plt.imshow(image_meta)
    plt.title(f'Meta Class {i}')
    plt.axis('off')

# Display 12 corresponding images from the Train folder
for i in range(12):
    plt.subplot(5, 12, 12 + i + 1)
    indices_classe = np.where(y_train == i)[0]
    plt.imshow(X_train[indices_classe[0]])
    plt.title(f'Train Class {i}')
    plt.axis('off')


# Display 12 enhanced images

for i in range(12):
    indices_classe = np.where(y_train == i)[0]
    plt.subplot(5, 12, 24 + i + 1)
    plt.imshow(images_enhancement(X_train, width=25, height=25, mode='RGB')[indices_classe[0]])
    plt.title(f' {i} - RGB')
    plt.axis('off')

for i in range(12):
    indices_classe = np.where(y_train == i)[0]
    plt.subplot(5, 12, 48 + i + 1)
    plt.imshow(images_enhancement(X_train, width=25, height=25, mode='RGB-HE')[indices_classe[0]])
    plt.title(f' {i} - RGB')
    plt.axis('off')

plt.tight_layout()
plt.show()



#########################################
    
# Save the dataSet

def save_h5_dataset(X_train, y_train, X_test, y_test, X_meta,y_meta, filename):
        
    # ---- Create h5 file
    with h5py.File(filename, "w") as f:
        f.create_dataset("x_train", data=X_train)
        f.create_dataset("y_train", data=y_train)
        f.create_dataset("x_test",  data=X_test)
        f.create_dataset("y_test",  data=y_test)
        f.create_dataset("x_meta",  data=X_meta)
        f.create_dataset("y_meta",  data=y_meta)
        
    # ---- done
    size=os.path.getsize(filename)/(1024*1024)
    print('Dataset : {:24s}  shape : {:22s} size : {:6.1f} Mo   (saved)'.format(filename, str(X_train.shape),size))






n_train = int(len(X_train) * scale)
n_test = int(len(X_test) * scale)

print('Parameters:')
print(f'Scale is: {scale}')
print(f'x_train length is: {n_train}')
print(f'x_test  length is: {n_test}')
print(f'output dir is: {output_dir}\n')

print('Running...')

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

for s in [24, 48]:
    for m in ['RGB', 'RGB-HE', 'L', 'L-LHE']:
        # A nice dataset name
        filename = f'{output_dir}/set-{s}x{s}-{m}.h5'
        print(f'Dataset: {filename}')

        # Enhancement
        x_train_new = images_enhancement(X_train[:n_train], width=s, height=s, mode=m)
        x_test_new = images_enhancement(X_test[:n_test], width=s, height=s, mode=m)
        x_meta_new = images_enhancement(X_meta, width=s, height=s, mode='RGB')

        # Save
        save_h5_dataset(x_train_new, y_train[:n_train], x_test_new, y_test[:n_test], x_meta_new, y_meta, filename)

x_train_new, x_test_new = 0, 0







