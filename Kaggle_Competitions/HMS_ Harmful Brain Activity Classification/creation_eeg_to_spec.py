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


group_electrodes = ['LL','LP','RP','RR']

electrodes = [['Fp1','F7','T3','T5','O1'],
         ['Fp1','F3','C3','P3','O1'],
         ['Fp2','F8','T4','T6','O2'],
         ['Fp2','F4','C4','P4','O2']]

directory_path = '/Users/jordanmoles/Documents/Programmes_Informatiques/Python/Projects/Kaggle_Competitions/hms-harmful-brain-activity-classification/EEG_Spectrograms/'
if not os.path.exists(directory_path):
    os.makedirs(directory_path)






USE_WAVELET ='db8'

# Denoising
def plot_eeg(df, moving_avg=1):
    fig, axs = plt.subplots(20, 1, figsize=(30, 15), sharex=True)
    for i, ax in enumerate(axs):
        ax.plot(df.iloc[:,i], color="black")
        ax.set_ylabel(df.columns[i], rotation=0)
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines[["top", "bottom", "left", "right"]].set_visible(False)


# Create a Median absolute deviation function
def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

# Create a denoising function
def denoise(x, wavelet='haar', level=1):
    ret = {key:[] for key in x.columns}
    
    for pos in x.columns:
        coeff = pywt.wavedec(x[pos], wavelet, mode="per")
        sigma = (1/0.6745) * maddest(coeff[-level])

        uthresh = sigma * np.sqrt(2*np.log(len(x)))
        coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])

        ret[pos]=pywt.waverec(coeff, wavelet, mode='per')
    
    return pd.DataFrame(ret)


# get eeg signal
GET_ROW = 0
row = df.iloc[GET_ROW]
df_eeg = pd.read_parquet(f'{train_eeg_path}{row.eeg_id}.parquet')


# Before denoising
eeg_offset = int( row.eeg_label_offset_seconds )
df_eeg = df_eeg.iloc[eeg_offset*200:(eeg_offset+50)*200]
plot_eeg(df_eeg)


# After denoising
eeg_denoised = denoise(df_eeg, wavelet="db8")
plot_eeg(eeg_denoised)



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


def spectrogram_from_eeg(parquet_path, display=False):
    
    # LOAD MIDDLE 50 SECONDS OF EEG SERIES
    eeg = pd.read_parquet(parquet_path)
    middle = (len(eeg)-10_000)//2
    eeg = eeg.iloc[middle:middle+10_000]
    
    # VARIABLE TO HOLD SPECTROGRAM
    img = np.zeros((128,256,4),dtype='float32')
    
    if display: plt.figure(figsize=(10,7))
    signals = []
    for k in range(4):
        COLS = electrodes[k]
        
        for kk in range(4):
        
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
            img[:,:,k] += mel_spec_db
                
        # AVERAGE THE 4 MONTAGE DIFFERENCES
        img[:,:,k] /= 4.0
        
        if display:
            plt.subplot(2,2,k+1)
            plt.imshow(img[:,:,k],aspect='auto',origin='lower')
            plt.title(f'EEG {eeg_id} - Spectrogram {group_electrodes[k]}')
            
    if display: 
        plt.show()
        plt.figure(figsize=(10,5))
        offset = 0
        for k in range(4):
            if k>0: offset -= signals[3-k].min()
            plt.plot(range(10_000),signals[k]+offset,label=group_electrodes[3-k])
            offset += signals[3-k].max()
        plt.legend()
        plt.title(f'EEG {eeg_id} Signals')
        plt.show()
        print(); print('#'*25); print()
        
    return img





# Create directory eeg_to_spectrograms
directory_path = '/Users/jordanmoles/Documents/Programmes_Informatiques/Python/Projects/Kaggle_Competitions/hms-harmful-brain-activity-classification/EEG_Spectrograms/'
if not os.path.exists(directory_path):
    os.makedirs(directory_path)


PATH = '/Users/jordanmoles/Documents/Programmes_Informatiques/Python/Projects/Kaggle_Competitions/hms-harmful-brain-activity-classification/train_eegs/'
DISPLAY = 4
EEG_IDS = df.eeg_id.unique()
all_eegs = {}

for i,eeg_id in tqdm(enumerate(EEG_IDS), total=len(EEG_IDS), desc="Processing EEG IDs"):
    if (i%100==0)&(i!=0): print(i,', ',end='')
        
    # CREATE SPECTROGRAM FROM EEG PARQUET
    img = spectrogram_from_eeg(f'{PATH}{eeg_id}.parquet', i<DISPLAY)
    
    # SAVE TO DISK
    if i==DISPLAY:
        print(f'Creating and writing {len(EEG_IDS)} spectrograms to disk... ',end='')
    np.save(f'{directory_path}{eeg_id}',img)
    all_eegs[eeg_id] = img
   
# SAVE EEG SPECTROGRAM DICTIONARY
np.save('eeg_specs',all_eegs)