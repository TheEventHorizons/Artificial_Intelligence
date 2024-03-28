import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import mne
from mne.time_frequency import tfr_morlet
import cv2
import os
from tqdm import tqdm


data_train_path = '/Users/jordanmoles/Documents/Programmes_Informatiques/Python/Projects/Kaggle_Competitions/hms-harmful-brain-activity-classification/train.csv'





##############################################################################################
#                                       INTRODUCTION
##############################################################################################


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



##############################################################################################
#                                       MONTAGES USED
##############################################################################################


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



directory_path = '/Users/jordanmoles/Documents/Programmes_Informatiques/Python/Projects/Kaggle_Competitions/hms-harmful-brain-activity-classification/EEG_Spectrograms/'
if not os.path.exists(directory_path):
    os.makedirs(directory_path)




# Select EEG
    
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


# Define a dictionnary of non-overlapping events of ten seconds
df_dict={}

for eeg_id in df['eeg_id'].unique():

    # Create a DataFrame for each 'eeg_id' and store it in the dictionary
    df_dict[f'df_{eeg_id}'] = df[df['eeg_id'] == eeg_id].copy()

    # Access the DataFrame for the current 'eeg_id'
    df_eeg = df_dict[f'df_{eeg_id}'].copy()

    # Create the non overlapping events
    events = df_eeg[['eeg_label_offset_seconds']]
    non_overlapping_events = pd.DataFrame(columns=events.columns)

    # Create the list of all eeg_label_offset_seconds
    list_eeg_label_offset_seconds = list(df_eeg['eeg_label_offset_seconds'])
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
    mask = df_eeg['eeg_label_offset_seconds'].isin(non_overlapping_mask)

    # Apply the mask to get non-overlapping events
    df_dict[f'df_{eeg_id}'] = df_eeg[mask]


# Create the non overlapping events
events = df_eeg[['eeg_label_offset_seconds']]
non_overlapping_events = pd.DataFrame(columns=events.columns)

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




##############################################################################################
#                                       CREATE OBJECTS
##############################################################################################

# Define a function which return a dictionnary of epochs based on eeeg_id, events and the montage 
def create_objects(eeg_id, events, event_ids, DIC_MONTAGE):
    
    # Read the corresponding eeg
    data = pd.read_parquet(f'/Users/jordanmoles/Documents/Programmes_Informatiques/Python/Projects/Kaggle_Competitions/hms-harmful-brain-activity-classification/train_eegs/{eeg_id}.parquet').copy()
    
    # Drop the EKG column
    data = data.drop(['EKG'], axis=1)
    
    # Compute pair differences and load it in a matrix
    dic_raw = {}
    dic_epoch={}

    for name, COLS in DIC_MONTAGE.items():
        raw=[] 
        ch_names = []
        for kk in range(len(COLS)-1):
            # COMPUTE PAIR DIFFERENCES
            ch_names.append(f'{COLS[kk]}-{COLS[kk+1]}')
            x = data[COLS[kk]].values - data[COLS[kk+1]].values
            # FILL NANS
            m = np.nanmean(x)
            if np.isnan(x).mean()<1: x = np.nan_to_num(x,nan=m)
            else: x[:] = 0
            # Put it in the matrix
            raw.append(x)
        ch_types = ['misc']*(len(ch_names)) 
        info = mne.create_info(ch_names, ch_types=ch_types, sfreq=200,verbose=False)
        raw_array = np.array(raw)
        raw_object = mne.io.RawArray(raw_array, info,verbose=False)
        dic_raw[name] = raw_object
        dic_epoch[name]= mne.Epochs(raw_object, events, event_id = event_ids, tmin=-5, tmax=5, preload=True, baseline=(None, 0),verbose=False)
    return dic_epoch


event_ids = {}

# Create event ids for this eeg
for activity in df_dict[f'df_{eeg_id}']['expert_consensus'].unique():
    event_ids[activity] = Target[activity]


# Create the events
events = df_dict[f'df_{eeg_id}'][['eeg_label_offset_seconds', 'expert_consensus']]
events.insert(1, 'New', 0)

# Map 'expert_consensus' values to numerical labels using the 'event_ids' dictionary
events.loc[:,'expert_consensus'] = events['expert_consensus'].map(event_ids)

# Define the sample time point where the event occurs
events.loc[:,'eeg_label_offset_seconds'] = (events['eeg_label_offset_seconds']+25)*200

# Convert the data frame into an array
events = events.values.astype(int)


# Plot the first epochs for all channels.
#create_objects(1628180742,events, event_ids, DIC_BANANA)['LL'][0].plot(n_epochs=4, events=True, picks = 'all', show_scrollbars=False, show_scalebars=False, scalings=300)
#plt.tight_layout()



##############################################################################################
#                                  GENERATING SPECTROGRAMS WITH MNE
##############################################################################################




def spectrogram_from_eeg(eeg_id, dic_epoch, DIC_MONTAGE_SPEC, NAME_MONTAGE, num_epoch, display=False):
    
    # Define the range of frequencies
    freq = np.arange(0.1, 20, 0.01)

    # Define the number of cycles
    n_cycles = freq/2
    
        
    #  Define the Morlet wavelet transform for the epoch num_epoch associated to the event NAME_MONTAGE
    power_seizure = tfr_morlet(dic_epoch[NAME_MONTAGE][num_epoch], freq, n_cycles = n_cycles, return_itc = False, picks='all', verbose=False)
    
    # Initialize the list to store the spectrograms
    spectrograms = []

    # Define the height and width of the resized image
    height = 128
    width = 256

    # Initialize the resized image
    img = np.zeros((height, width))
    
    # Compute the spectrograms for each channel
    for title in DIC_MONTAGE_SPEC[NAME_MONTAGE]:
        fig, ax = plt.subplots(figsize=(12,6))
        power_seizure.plot(picks=title, title=title, axes=ax, show=False, verbose=False)
        spectrogram = ax.images[0].get_array().astype(np.float32)  # Get the data array of the spectrogram
        spectrogram = (spectrogram + 40) / 40  # Normalize the spectrogram
        spectrograms.append(spectrogram)
        plt.close(fig)


    # Resize the image to a lower resolution
    img = cv2.resize(sum(spectrograms)/len(DIC_MONTAGE_SPEC[NAME_MONTAGE]), (width, height), interpolation=cv2.INTER_AREA)
    # Display the resized spectrogram
    if display:
        plt.figure()
        plt.imshow(img, aspect='auto', origin='lower')
        plt.colorbar(label='Power')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.title(f'EEG {eeg_id} {NAME_MONTAGE} {num_epoch}')
        plt.show()
        
    return img


# spectrogram_from_eeg(1628180742, create_objects(1628180742,events, event_ids, DIC_BANANA),DIC_BANANA_SPEC, NAME_MONTAGE='LL', num_epoch=0, display=True)


EEG_IDS = df.eeg_id.unique()
#EEG_IDS = [1628180742,2277392603]
all_eegs = {}




print('Starting the creation of the dictionnary')

for i,eeg_id in tqdm(enumerate(EEG_IDS)):
    if (i%100==0)&(i!=0): print(i,', ',end='')
    event_ids = {}
    print(f'Starting the creation of spectrograms associated to the eeg_id {eeg_id}')
    # Create event ids for this eeg
    for activity in df_dict[f'df_{eeg_id}']['expert_consensus'].unique():
        event_ids[activity] = Target[activity]


    # Create the events
    events = df_dict[f'df_{eeg_id}'][['eeg_label_offset_seconds', 'expert_consensus']]
    events.insert(1, 'New', 0)

    # Map 'expert_consensus' values to numerical labels using the 'event_ids' dictionary
    events.loc[:,'expert_consensus'] = events['expert_consensus'].map(event_ids)

    # Define the sample time point where the event occurs
    events.loc[:,'eeg_label_offset_seconds'] = (events['eeg_label_offset_seconds']+25)*200

    # Convert the data frame into an array
    events = events.values.astype(int)    

    # Create an array to store the spectrograms for each NAME_MONTAGE and each epoch
    eeg_spectrograms = np.zeros((128, 256, len(DIC_BANANA.keys()), events.shape[0]), dtype=np.float32)

    # Create an index counting the number of NAME_MONTAGE
    idx = 0  

    # Create the necessary objects to generate the spectrograms
    create_objects(eeg_id, events, event_ids, DIC_BANANA)

    # Loop over each montage
    for NAME in list(DIC_BANANA.keys()):
        # Loop over each epoch
        for j in range(events.shape[0]):   
            # Generate the spectrogram
            img = spectrogram_from_eeg(eeg_id, create_objects(eeg_id, events, event_ids, DIC_BANANA), DIC_BANANA_SPEC, NAME_MONTAGE=NAME, num_epoch=j, display=False).astype(np.float32)

            # Add the spectrogram to the array
            eeg_spectrograms[:, :, idx, j] = img
            np.save(f'{directory_path}{eeg_id}',img)
            
        # Increment the index for montages
        idx+=1

    # Add the array of spectrograms for this EEG to the global dictionary
    all_eegs[eeg_id] = eeg_spectrograms
    print(f'Spectrograms associated to the eeg_id {eeg_id} saved')

# Save the dictionary containing all the spectrograms
np.save('eeg_specs', all_eegs)

print('Dictionnary saved')