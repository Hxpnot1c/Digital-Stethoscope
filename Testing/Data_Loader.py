import librosa
import numpy as np
import pandas as pd
import torch

DATASET = 'Res/DigiScope Dataset/'
SAMPLE_RATE = 10000

def load_WAV(filename):
    data, sr = librosa.load(DATASET + filename, sr=22050, offset=0)
    resampled_data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLE_RATE)
    return resampled_data

def gen_features(filename):
    data = load_WAV(filename)
    features = np.array([])

    mean_zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    features = np.hstack((features, mean_zero_crossing_rate))

    stft_data = np.abs(librosa.stft(data))
    chroma_mean = np.mean(librosa.feature.chroma_stft(S=stft_data, sr=SAMPLE_RATE).T, axis=0)
    features = np.hstack((features, chroma_mean))

    mfcc_data = np.mean(librosa.feature.mfcc(y=data, sr=SAMPLE_RATE).T, axis=0)
    features = np.hstack((features, mfcc_data))
    
    rms_data = np.mean(librosa.feature.rms(y=data).T, axis=0)
    features = np.hstack((features, rms_data))

    mel_data = np.mean(librosa.feature.melspectrogram(y=data, sr=SAMPLE_RATE).T, axis=0)
    features = np.hstack((features, mel_data))

    return features

labels = pd.read_csv(DATASET + 'training_labels.csv')
x_train = []
y_train = []
for _, row in labels.iterrows():
    features = gen_features(row['filename'])
    x_train.append(features)
    y_train.append(row['label'])

train_df = pd.DataFrame(x_train)
train_df['class'] = y_train
train_df.to_csv(DATASET + 'train_data.csv', index=False)

def one_hot_encode(labels, classes):
    encoded_labels = np.zeros((labels.size, classes))
    encoded_labels[np.arange(labels.size), labels] = 1
    return encoded_labels

one_hot_y = one_hot_encode(np.array(y_train), 3)
class AudioDataset():
    def __init__(self, data_csv):
        self.data = pd.read_csv(DATASET + data_csv)
        self.encoded_labels = one_hot_encode(np.array(self.data.iloc[:,-1]), 3)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        features = torch.tensor(self.data.iloc[index,:-1].values)
        label = torch.tensor(self.encoded_labels[index])
        return features, label
    
data = AudioDataset('train_data.csv')