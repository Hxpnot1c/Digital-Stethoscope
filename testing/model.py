import torch.nn as nn

# Class for our deep convolutional neural network
class ConvolutionalNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialises first convolutional layer
        # Takes tensor input of shape (batch_num, 5, 126)
        # Outputs tensor of shape (batch_num, 256, 62)
        # Has a dropout layer to reduce overfitting
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=256, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=256),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Dropout(0.2)
        )

        # Initialises second convolutional layer
        # Takes tensor input of shape (batch_num, 256, 62)
        # Outputs tensor of shape (batch_num, 60, 20)
        # Has a dropout layer to reduce overfitting
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=64, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            nn.Dropout(0.2)
        )

        # Initialises third convolutional layer
        # Takes tensor input of shape (batch_num, 60, 20)
        # Outputs tensor of shape (batch_num, 16, 6)
        # Has a dropout layer to reduce overfitting
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=16, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            nn.Dropout(0.2)
        )

        # Initialises flattening layer
        # Flattens input of shape (batch_num, 16, 6) to output of shape (batch_num, 96)
        self.flatten = nn.Flatten()

        # Initialises first linear layer
        # Takes tensor input of shape (batch_num, 96)
        # Outputs tensor of shape (batch_num, 8)
        self.linear1 = nn.Linear(in_features=16*6, out_features=8)
        self.linear_activation1 = nn.ReLU() # Applies ReLU activation function after first linear layer

        # Initialises second linear layer
        # Takes tensor input of shape (batch_num, 8)
        # Outputs tensor of shape (batch_num, 3) as one hot encoded labels are of size 3
        self.linear2 = nn.Linear(in_features=8, out_features=3)
        self.output = nn.LogSoftmax(dim=-1) # Applies LogSoftmax activation function after second linear layer to get final outputs
        
    
    def forward(self, input):
        # Performs forward propagation on input
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear_activation1(x)
        logits = self.linear2(x)
        output = self.output(logits)

        return output
    