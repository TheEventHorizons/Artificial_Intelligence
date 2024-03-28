import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import mne
from mne.time_frequency import tfr_morlet
import cv2
import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

data_train_path = '/kaggle/input/hms-harmful-brain-activity-classification/train.csv'


# Read the data
data = pd.read_csv(data_train_path)

# Copy the Data
df = data.copy()

# Observe few lines 
print(df.head())


# Shape of the data
print('The shape of df is:', df.shape)


# Create Target
Target = {'Seizure':0, 'LPD':1, 'GPD':2, 'LRDA':3, 'GRDA':4, 'Other':5}



# Montages Used


NAMES_BANANA = ['LL','LP','RP','RR','BC']

FEATS_BANANA = [['Fp1','F7','T3','T5','O1'],
                ['Fp1','F3','C3','P3','O1'],
                ['Fp2','F8','T4','T6','O2'],
                ['Fp2','F4','C4','P4','O2'],
                ['Fz','Cz','Pz']]

DIC_BANANA = {}
for name, feat in zip(NAMES_BANANA,FEATS_BANANA):
    DIC_BANANA[name]=feat


DIC_BANANA_SPEC = {}
for name, feat in zip(NAMES_BANANA, FEATS_BANANA):
    ch_names = []
    for kk in range(len(feat)-1):
        # COMPUTE PAIR DIFFERENCES
        ch_names.append(f'{feat[kk]}-{feat[kk+1]}')
    DIC_BANANA_SPEC[name] = ch_names

print(DIC_BANANA_SPEC)



directory_path = 'EEG_Spectrograms/'
if not os.path.exists(directory_path):
    os.makedirs(directory_path)




# Select EEG
    
# Adding a new column 'total_evaluators' that sums up the six specified columns
df.loc[:,'total_evaluators'] = df[['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']].sum(axis=1)
print(df.head())