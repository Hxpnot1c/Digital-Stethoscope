import numpy as np
import librosa
import pathlib
import torch
from time import sleep
import pandas as pd
from model import ConvolutionalNN

root_dir = pathlib.Path(__file__).resolve().parent

df = pd.DataFrame([[-1, -1]])
df.to_csv(root_dir / 'inference.csv', index=False)

device = torch.device('cpu')
model = ConvolutionalNN().to(device)
model.load_state_dict(torch.load(root_dir / 'model.pth', map_location=device))
model.eval()

def gen_features():
    SAMPLE_RATE = 10000
    original_data, sr = librosa.load(root_dir / 'audio_data.wav', sr=1000, offset=0)
    data = librosa.resample(original_data, orig_sr=sr, target_sr=SAMPLE_RATE)

    mean_zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    mean_zero_crossing_rate = np.pad(mean_zero_crossing_rate, (64, 63))
    features = np.array(mean_zero_crossing_rate)

    stft_data = np.abs(librosa.stft(data))
    chroma_mean = np.mean(librosa.feature.chroma_stft(S=stft_data, sr=SAMPLE_RATE).T, axis=0)
    chroma_mean = np.pad(chroma_mean, (58, 58))
    features = np.vstack((features, chroma_mean))

    mfcc_data = np.mean(librosa.feature.mfcc(y=data, sr=SAMPLE_RATE).T, axis=0)
    mfcc_data = np.pad(mfcc_data, (54, 54))
    features = np.vstack((features, mfcc_data))
    
    rms_data = np.mean(librosa.feature.rms(y=data).T, axis=0)
    rms_data = np.pad(rms_data, (64, 63))
    features = np.vstack((features, rms_data))

    mel_data = np.mean(librosa.feature.melspectrogram(y=data, sr=SAMPLE_RATE).T, axis=0)
    features = np.vstack((features, mel_data))

    return features

def one_hot_encode(label):
        encoded_labels = [0, 0, 0]
        encoded_labels[label] = 1
        return torch.tensor(encoded_labels, device=device)

input = torch.tensor(gen_features(), device=device).unsqueeze(0)
loss_fucntion = torch.nn.CrossEntropyLoss()
while True:
    sleep(2)
    if not torch.all(input.eq(new_data := torch.tensor(gen_features(), device=device).unsqueeze(0))):
        input = new_data
        output = model(input.float())
        _, prediction = torch.max(output, 1)
        prediction = int(prediction)
        #confidence = (1-float(torch.nn.CrossEntropyLoss(output.float(), input.float()).detach)) * 100
        confidence = 1-abs(loss_fucntion(output.float(), one_hot_encode(prediction).unsqueeze(0).float()).item())
        print(f'Diagnosis: {prediction}\nConfidence: {confidence*100:.1f}')
        df = pd.DataFrame([[prediction, confidence]])
        df.to_csv(root_dir / 'inference.csv', index=False)
