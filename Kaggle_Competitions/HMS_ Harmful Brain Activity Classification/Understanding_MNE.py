'''
MNE-Python stands as a robust Python library tailored for the intricate tasks of processing, analyzing, and visualizing neurophysiological data, with a specific focus on magnetoencephalography (MEG) and electroencephalography (EEG) data. Engineered with a comprehensive array of tools, this library empowers researchers and scientists to navigate the complexities of time-series neuroimaging data. By leveraging MNE-Python, professionals can conduct diverse analyses, unlocking profound insights into the intricate workings of brain activity.

Our exploration will delve into the practical application of this module using the HMS dataset, providing a hands-on experience to unravel its potential and functionalities in real-world scenarios.



import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import mne


data_train_path = '/kaggle/input/hms-harmful-brain-activity-classification/train.csv'



# Read the data
data = pd.read_csv(data_train_path)

# Copy the Data
df = data.copy()

# Observe few lines 
print(df.head())


# Shape of the data
print('The shape of df is:', df.shape)

In order to understand the command of MNE, we will take the eeg_ids which has 5 expert consensus (the maximum) 'eeg_id'== 1460778765.

# Grouping the DataFrame by 'eeg_id' and counting the number of unique 'expert_consensus' values for each group
counts = df.groupby('eeg_id')['expert_consensus'].nunique()

# Select the eeg_id that have more than 4 unique expert consensus
eeg_id_with_multiple_consensus = counts[counts > 4].index.tolist()
print('Here is the list of eeg_id with more that 4 unique expert consensus' ,eeg_id_with_multiple_consensus)

# Take the eeg_id which as five expert consensus
df_multiple = df[df['eeg_id']==1460778765]
print(df_multiple)


First we have to create essential MNE data structures (Raws, Events, Epochs, Evoked) with something called measurement information.

# Read the corresponding eeg
data_eeg_path = '/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/21379701.parquet'
data = pd.read_parquet(data_eeg_path).copy()

1. **Create measurement information**

This data structure behaves like a dictionary. It contains all metadata that is available for a recording. However, its keys are restricted to those provided by the FIF format specification, so new entries should not be manually added. (source MNE website)

In our case, since we do not have such file we create the info from scratch and we're going to use create_info. It contains the list of columns in the eeg, the type and the frequency which in our case is 200 samples per second.

Remark: the available types are ‘ecg’, ‘bio’, ‘stim’, ‘eog’, ‘misc’, ‘seeg’, ‘dbs’, ‘ecog’, ‘mag’, ‘eeg’, ‘ref_meg’, ‘grad’, ‘emg’, ‘hbr’ ‘eyetrack’ or ‘hbo’.

# Define the channel names
ch_names = data.columns.tolist()

# Define the channel types
ch_types = ['misc']*len(data.columns) 

# Create the measurement information
info = mne.create_info(ch_names, ch_types=ch_types, sfreq=200)
print(info)


2. **Create the Raw object**

In MNE-Python, the Raw object is a fundamental data structure used to represent continuous neurophysiological data, such as EEG (electroencephalography) or MEG (magnetoencephalography) recordings. It is part of the MNE core data structures and is designed to store raw sensor data along with associated metadata.

- Raw Data: It holds the raw sensor data as a 2D NumPy array, where each row corresponds to a sensor (channel), and each column corresponds to a time point.

- Metadata: It's the measurement information which includes information about the recording, such as sensor locations, sampling frequency, channel names, and other relevant information.

- Methods for Data Analysis: The Raw object provides various methods for basic data analysis tasks, such as filtering, resampling, and applying annotations.

- Visualization: MNE-Python includes visualization tools to plot the raw data, allowing users to inspect and analyze the recordings.

To initialize a Raw object from the ground up, we begin by converting our DataFrame into a NumPy array and then transpose it.


# Create the numpy array and transpose it
data_values = data.values.T

# Create the Raw object
raw = mne.io.RawArray(data_values, info)

Now we can plot the raw object. To do this, we define a scaling object and the duration. The rest of the commands is used for visualization convenience.


# Define a scaling object
scalings = {'misc': 200} 

# Define the duration of the eeg in second
duration = df['eeg_label_offset_seconds'].max()+50
print(duration)

# Plot the raw object
raw.plot(show_scrollbars=False, show_scalebars=False, duration= duration, scalings=scalings)


3. **Create the Epochs object**


Epochs objects provide a means of representing continuous data as a collection of time-locked trials, stored in an array of shape (n_events, n_channels, n_times). They are valuable for various statistical methods in neuroscience and facilitate a quick overview of what occurs during a trial.

To epoch the data, event markers are required, typically stored in the raw object within a channel known as the stimulus channel. In our scenario, we need to create these events, which is essentially an array in the format (time in samples, zero, trigger).


# Create event ids
event_ids = {'Seizure':0, 'LPD':1, 'GPD':2, 'LRDA':3, 'GRDA':4, 'Other':5}

# Create the events
events = df_multiple[['eeg_label_offset_seconds', 'expert_consensus']]
events.insert(1, 'New', 0)
print(events)

# Map 'expert_consensus' values to numerical labels using the 'event_ids' dictionary
events.loc[:,'expert_consensus'] = events['expert_consensus'].map(event_ids)

# Events must be in sample
events.loc[:,'eeg_label_offset_seconds'] = events['eeg_label_offset_seconds']*200

# Convert the data frame into an array
events = events.values.astype(int)
#print(events)


# Plot the events
mne.viz.plot_events(events[:])

Now, we generate epochs, representing time windows around events. Setting the starting time `tmin=-5` and ending time `tmax=5` creates a time window of 10 seconds around each event.

# Create the epochs
epochs = mne.Epochs(raw, events, tmin=-5, tmax=5, preload=True, baseline=(None, 0))
print(epochs.event_id)

# Plot the epochs for all channels.
epochs.plot(n_epochs=events.shape[0], events=True, picks = 'all', show_scrollbars=False, show_scalebars=False, scalings=300)

Many epochs overlap, likely due to closely spaced or identical events. To address the issue, we will select events that are spaced at least 10 seconds apart (non-overlapping events).

# Create the non overlapping events
events = df_multiple[['eeg_label_offset_seconds', 'expert_consensus']]
non_overlapping_events = pd.DataFrame(columns=events.columns)

# Create the list of all eeg_label_offset_seconds
list_eeg_label_offset_seconds = list(df_multiple['eeg_label_offset_seconds'])
print(list_eeg_label_offset_seconds)

# Generate a list of eeg_label_offset_seconds with a minimum separation of 10 seconds
new_list = []
current_offset = 0
min_distance = 10

while current_offset <= max(list_eeg_label_offset_seconds):
    new_list.append(current_offset)
    next_offset = next((x for x in list_eeg_label_offset_seconds if x >= current_offset + min_distance), None)
    if next_offset is None:
        break
    current_offset = next_offset
    
print(new_list)

# Create the non overlapping events
events = df_multiple[['eeg_label_offset_seconds', 'expert_consensus']]
non_overlapping_events = pd.DataFrame(columns=events.columns)

# Create a mask using the new_list
mask = events['eeg_label_offset_seconds'].isin(new_list)

# Apply the mask to get non-overlapping events
non_overlapping_events = events[mask]

# print the non_overlapping_events
non_overlapping_events.insert(1, 'New', 0)
print(non_overlapping_events)

# Map 'expert_consensus' values to numerical labels using the 'event_ids' dictionary
non_overlapping_events.loc[:,'expert_consensus'] = non_overlapping_events['expert_consensus'].map(event_ids)

non_overlapping_events.loc[:,'eeg_label_offset_seconds']=non_overlapping_events.loc[:,'eeg_label_offset_seconds']*200

# Convert the data frame into an array
non_overlapping_events = non_overlapping_events.values.astype(int)
#print(non_overlapping_events)

# Plot the events
mne.viz.plot_events(non_overlapping_events[:])

# Create the epochs
epochs = mne.Epochs(raw, non_overlapping_events, tmin=-5, tmax=5, preload=True, baseline=(None, 0))
print(epochs.event_id)

# Plot the epochs for all channels.
epochs.plot(n_epochs=non_overlapping_events.shape[0], events=True, picks = 'all', show_scrollbars=False, show_scalebars=False, scalings=300)
'''