import torch.nn as nn


class ConvolutionalNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=256, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=256),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Dropout(0.2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=64, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            nn.Dropout(0.2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=16, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            nn.Dropout(0.2)
        )

        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(in_features=16*6, out_features=8)

        self.linear_activation1 = nn.ReLU()

        self.linear2 = nn.Linear(in_features=8, out_features=3)
            
        self.output = nn.LogSoftmax(dim=-1)
        
    
    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear_activation1(x)
        logits = self.linear2(x)
        output = self.output(logits)

        return output
    