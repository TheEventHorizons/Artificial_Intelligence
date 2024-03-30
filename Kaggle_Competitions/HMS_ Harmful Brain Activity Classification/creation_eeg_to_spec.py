import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import os
import pywt

import librosa

from sklearn.pipeline import make_pipeline



data_train_path = '/Users/jordanmoles/Documents/Programmes_Informatiques/Python/Projects/Kaggle_Competitions/hms-harmful-brain-activity-classification/train.csv'
train_eeg_path = '/Users/jordanmoles/Documents/Programmes_Informatiques/Python/Projects/Kaggle_Competitions/hms-harmful-brain-activity-classification/train_eegs/'


# Display the max row and the max columns
pd.set_option('display.max_row',111)
#pd.set_option('display.max_columns',111)

# Read the data
data = pd.read_csv(data_train_path)

# Copy the Data
df = data.copy()


# Observe few lines 
print(df.head())

# Shape of the data
print('The shape of df is:', df.shape)

# Create columns
Column_name = list(df.columns)




############################################################################################################################################################################################
#                                                                                          The Bipolar Double Banana Montage
############################################################################################################################################################################################


NAMES = ['LL','LP','RP','RR','BC']

FEATS = [['Fp1','F7','T3','T5','O1'],
         ['Fp1','F3','C3','P3','O1'],
         ['Fp2','F8','T4','T6','O2'],
         ['Fp2','F4','C4','P4','O2'],
         ['Fz','Cz','Pz']]

directory_path = '/Users/jordanmoles/Documents/Programmes_Informatiques/Python/Projects/Kaggle_Competitions/hms-harmful-brain-activity-classification/EEG_Spectrograms/'
if not os.path.exists(directory_path):
    os.makedirs(directory_path)


############################################################################################################################################################################################
#                                                                                                   Select eeg
############################################################################################################################################################################################

# Adding a new column 'total_evaluators' that sums up the six specified columns
df.loc[:,'total_evaluators'] = df[['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']].sum(axis=1)
print(df.head())


# Modifying the previous code to add an additional column 'consensus_column' to 'df'

# Finding the column with the largest number for each row and storing the value in 'consensus'
df.loc[:,'consensus'] = df[['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']].max(axis=1)

# Identifying the column name that corresponds to the max value for each row
df.loc[:,'consensus_column'] = df[['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']].idxmax(axis=1)

# create a new column that shows the percentage agreement
df.loc[:,'row_agreement'] = df['consensus']/df['total_evaluators']

# Delete rows for which the DataFrame has less than 30% of row_agreement
df = df[df['row_agreement']>0.3]
print(df)

# Create the list of all eeg_label_offset_seconds
list_eeg_label_offset_seconds = list(df['eeg_label_offset_seconds'])
#print(list_eeg_label_offset_seconds)

# Generate a list of eeg_label_offset_seconds with a minimum separation of 10 seconds   
non_overlapping_mask=[]
current_offset = 0
min_distance = 10

while current_offset <= max(list_eeg_label_offset_seconds):
    non_overlapping_mask.append(current_offset)
    next_offset = next((x for x in list_eeg_label_offset_seconds if x >= current_offset + min_distance), None)
    if next_offset is None:
        break
    current_offset = next_offset

# Create a mask using the non_overlapping_mask
mask = df['eeg_label_offset_seconds'].isin(non_overlapping_mask)

df = df[mask]
print(df)


print(df.shape[0])
print(len(df[df['eeg_id']==f'{1628180742}']['eeg_label_offset_seconds'].tolist()))
print(len(df[df['eeg_id']==1628180742]['eeg_label_offset_seconds'].tolist()))




############################################################################################################################################################################################
#                                                                          Optional Signal Denoising with Wavelet transform
############################################################################################################################################################################################




USE_WAVELET ='db8'

# DENOISE FUNCTION
def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def denoise(x, wavelet='haar', level=1):    
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1/0.6745) * maddest(coeff[-level])

    uthresh = sigma * np.sqrt(2*np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])

    ret=pywt.waverec(coeff, wavelet, mode='per')
    
    return ret




############################################################################################################################################################################################
#                                                                          Create Spectrograms with Librosa
############################################################################################################################################################################################
import librosa

def spectrogram_from_eeg(eeg_id, display=False):
    
    # VARIABLE TO HOLD SPECTROGRAM
    img = np.zeros((128,256,len(NAMES), len(df[df['eeg_id']==eeg_id]['eeg_label_offset_seconds'].tolist())),dtype='float32')
    idx = 0
    for offset_seconds in df[df['eeg_id']==eeg_id]['eeg_label_offset_seconds'].tolist():
        # LOAD MIDDLE 10 SECONDS OF EEG SERIES
        eeg = pd.read_parquet(f'/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/{eeg_id}.parquet')
        starting_time = (offset_seconds+20)*200 
        ending_time = (offset_seconds+30)*200
        eeg = eeg.loc[starting_time:ending_time]


        if display: plt.figure(figsize=(10,7))
        signals = []
        for k in range(len(NAMES)):
            COLS = FEATS[k]
            for kk in range(len(COLS)-1):

                # COMPUTE PAIR DIFFERENCES
                x = eeg[COLS[kk]].values - eeg[COLS[kk+1]].values

                # FILL NANS
                m = np.nanmean(x)
                if np.isnan(x).mean()<1: x = np.nan_to_num(x,nan=m)
                else: x[:] = 0

                # DENOISE
                if USE_WAVELET:
                    x = denoise(x, wavelet=USE_WAVELET)
                signals.append(x)

                # RAW SPECTROGRAM
                mel_spec = librosa.feature.melspectrogram(y=x, sr=200, hop_length=len(x)//256, 
                      n_fft=1024, n_mels=128, fmin=0, fmax=20, win_length=128)

                # LOG TRANSFORM
                width = (mel_spec.shape[1]//32)*32
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[:,:width]

                # STANDARDIZE TO -1 TO 1
                mel_spec_db = (mel_spec_db+40)/40 
                img[:,:,k,idx] += mel_spec_db

            # AVERAGE THE 5 MONTAGE DIFFERENCES
            img[:,:,k,idx] /= len(COLS)

            if display:
                plt.subplot(3,2,k+1)
                plt.imshow(img[:,:,k,idx],aspect='auto',origin='lower')
                plt.title(f'EEG {eeg_id} - Spectrogram {NAMES[k]} - Number {idx}')
                plt.show()
        idx +=1
        
    return img




PATH = '/Users/jordanmoles/Documents/Programmes_Informatiques/Python/Projects/Kaggle_Competitions/hms-harmful-brain-activity-classification/train_eegs/'
DISPLAY = 4
EEG_IDS = df.eeg_id.unique()
all_eegs = {}

for i,eeg_id in enumerate(EEG_IDS):
    if (i%100==0)&(i!=0): print(i,', ',end='')
        
    # CREATE SPECTROGRAM FROM EEG PARQUET
    img = spectrogram_from_eeg(eeg_id, i<DISPLAY)
    
    # SAVE TO DISK
    if i==DISPLAY:
        print(f'Creating and writing {df.shape[0]} spectrograms to disk... ',end='')
    np.save(f'{directory_path}{eeg_id}',img)
    all_eegs[eeg_id] = img
   
# SAVE EEG SPECTROGRAM DICTIONARY
np.save('eeg_specs',all_eegs)


