import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import pathlib

# Load filenames and labels for data before upsampling
DATASET_DIR = pathlib.Path('res/Digiscope Dataset/')
annotations = pd.read_csv(DATASET_DIR.joinpath('training_labels.csv'))


# Iterate through data to get arrays of extrasystole and murmur filenames
extrasystole = []
murmur = []
for _, row in annotations.iterrows():
    if row['label'] == 1:
        extrasystole.append(row)

    elif row['label'] == 2:
        murmur.append(row)
extrastole = np.array(extrasystole)[:, 0]
murmur = np.array(murmur)[:, 0]


# Determine scale factors to upsample extrasystole and murmur data by
EXTRASYSTOLE_SF = int(np.round(536 / 46))
MURMUR_SF = int(np.round(536 / 95))


# Iterate through extrasystole files and make scale factor - 1 new synthetic files for each pre-existing file
extrastole_files = []
for i in range(EXTRASYSTOLE_SF-1):
    for filename in extrastole:
        data, sr = librosa.load(DATASET_DIR.joinpath(filename), sr=22050, offset=0)
        resampled_data = librosa.resample(data, orig_sr=sr, target_sr=1000)
        noise = 0.009 * np.random.uniform() * np.max(resampled_data)
        new_data = resampled_data + noise * np.random.normal(size=resampled_data.shape[0])
        new_filename = filename.rsplit('.', 1)[0] + '_synthetic_' + str(i) + '.wav'
        sf.write(DATASET_DIR.joinpath(new_filename), new_data, 1000)
        extrastole_files.append(new_filename)
annotations = pd.concat([annotations, pd.DataFrame(data={'filename': extrastole_files, 'label': list(np.full(len(extrastole_files), 1))})])

# Iterate through murmur files and make scale factor - 1 new synthetic files for each pre-existing file
murmur_files = []
for i in range(MURMUR_SF-1):
    for filename in murmur:
        data, sr = librosa.load(DATASET_DIR.joinpath(filename), sr=22050, offset=0)
        resampled_data = librosa.resample(data, orig_sr=sr, target_sr=1000)
        noise = 0.009 * np.random.uniform() * np.max(resampled_data)
        new_data = resampled_data + noise * np.random.normal(size=resampled_data.shape[0])
        new_filename = filename.rsplit('.', 1)[0] + '_synthetic_' + str(i) + '.wav'
        sf.write(DATASET_DIR.joinpath(new_filename), new_data, 1000)
        murmur_files.append(new_filename)
annotations = pd.concat([annotations, pd.DataFrame(data={'filename': murmur_files, 'label': list(np.full(len(murmur_files), 2))})])


# Saves new labelled data with upsampling to upsampled_training_labels.csv
annotations.to_csv(DATASET_DIR.joinpath('upsampled_training_labels.csv'), index=False)
