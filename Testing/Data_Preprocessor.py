import librosa
import numpy as np
import pandas as pd
import torch
import pathlib


class AudioDataset():
    def __init__(self, DATASET_DIR, annotations_file):
        self.DATASET_DIR = pathlib.Path(DATASET_DIR)
        annotations_file = self.DATASET_DIR.joinpath(annotations_file)
        self.annotations = pd.read_csv(annotations_file)
        self.encoded_labels = self.one_hot_encode(np.array(self.annotations['label']), 3)
        self.SAMPLE_RATE = 10000
    

    def __len__(self):
        return len(self.annotations)
    

    def __getitem__(self, index):
        file_dir = self.annotations.iloc[index, 0]
        features = torch.tensor(self.gen_features(file_dir))
        label = torch.tensor(self.encoded_labels[index])
        return features, label
    

    def one_hot_encode(self, labels, classes):
        encoded_labels = np.zeros((labels.size, classes))
        encoded_labels[np.arange(labels.size), labels] = 1
        return encoded_labels


    def load_WAV(self, filename):
        file_dir = self.DATASET_DIR.joinpath(filename)
        data, sr = librosa.load(file_dir, sr=22050, offset=0)
        resampled_data = librosa.resample(data, orig_sr=sr, target_sr=self.SAMPLE_RATE)
        return resampled_data
    

    def gen_features(self, filename):
        data = self.load_WAV(filename)
        features = np.array([])

        mean_zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
        features = np.hstack((features, mean_zero_crossing_rate))

        stft_data = np.abs(librosa.stft(data))
        chroma_mean = np.mean(librosa.feature.chroma_stft(S=stft_data, sr=self.SAMPLE_RATE).T, axis=0)
        features = np.hstack((features, chroma_mean))

        mfcc_data = np.mean(librosa.feature.mfcc(y=data, sr=self.SAMPLE_RATE).T, axis=0)
        features = np.hstack((features, mfcc_data))
        
        rms_data = np.mean(librosa.feature.rms(y=data).T, axis=0)
        features = np.hstack((features, rms_data))

        mel_data = np.mean(librosa.feature.melspectrogram(y=data, sr=self.SAMPLE_RATE).T, axis=0)
        features = np.hstack((features, mel_data))

        return features

DataTest = AudioDataset('Res/DigiScope Dataset/', 'training_labels.csv')
print(DataTest.__getitem__(0))