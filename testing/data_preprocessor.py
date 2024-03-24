import librosa
import numpy as np
import pandas as pd
import torch
import pathlib


class AudioDataset():
    '''This class works with the pytorch DataLoader to load training and test data from res/DigiScope_Dataset to use to train and test the model'''

    def __init__(self, DATASET_DIR, annotations_file):
        # Initialises class
        self.DATASET_DIR = pathlib.Path(DATASET_DIR)
        annotations_file = self.DATASET_DIR.joinpath(annotations_file)
        self.annotations = pd.read_csv(annotations_file)
        self.encoded_labels = self.one_hot_encode(np.array(self.annotations['label']), 3)
        self.SAMPLE_RATE = 10000
    

    def __len__(self):
        # This is one of the required functions
        # Returns length of dataset
        return len(self.annotations)
    

    def __getitem__(self, index):
        # This is one of the required functions
        # Returns a datapoint with its label for a given index
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        file_dir = self.annotations.iloc[index, 0]
        features = torch.tensor(self.gen_features(file_dir), device=device)
        label = torch.tensor(self.encoded_labels[index], device=device)
        return features, label
    

    def one_hot_encode(self, labels, classes):
        # One hot encodes label data to use to train the model
        encoded_labels = np.zeros((labels.size, classes))
        encoded_labels[np.arange(labels.size), labels] = 1
        return encoded_labels


    def load_WAV(self, filename):
        # Loads .wav file and resamples to 1000Hz and then back to 10000Hz to mimic data collected at 1000Hz that is then resampled to 10000Hz like our stethoscope data
        file_dir = self.DATASET_DIR.joinpath(filename)
        data, sr = librosa.load(file_dir, sr=22050, offset=0)
        resampled_data = librosa.resample(data, orig_sr=sr, target_sr=1000)
        resampled_data = librosa.resample(resampled_data, orig_sr=1000, target_sr=self.SAMPLE_RATE)
        return resampled_data
    

    def gen_features(self, filename):
        # Generates features such as zero crossing rate, chroma stft, mfcc, rms and mel spectrogram from audio data
        data = self.load_WAV(filename)

        mean_zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
        mean_zero_crossing_rate = np.pad(mean_zero_crossing_rate, (64, 63))
        features = np.array(mean_zero_crossing_rate)

        stft_data = np.abs(librosa.stft(data))
        chroma_mean = np.mean(librosa.feature.chroma_stft(S=stft_data, sr=self.SAMPLE_RATE).T, axis=0)
        chroma_mean = np.pad(chroma_mean, (58, 58))
        features = np.vstack((features, chroma_mean))

        mfcc_data = np.mean(librosa.feature.mfcc(y=data, sr=self.SAMPLE_RATE).T, axis=0)
        mfcc_data = np.pad(mfcc_data, (54, 54))
        features = np.vstack((features, mfcc_data))
        
        rms_data = np.mean(librosa.feature.rms(y=data).T, axis=0)
        rms_data = np.pad(rms_data, (64, 63))
        features = np.vstack((features, rms_data))

        mel_data = np.mean(librosa.feature.melspectrogram(y=data, sr=self.SAMPLE_RATE).T, axis=0)
        features = np.vstack((features, mel_data))

        return features
