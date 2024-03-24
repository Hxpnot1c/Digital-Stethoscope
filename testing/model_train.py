from data_preprocessor import AudioDataset
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch
from torchsummary import summary
from tqdm import tqdm
from model import ConvolutionalNN


# Loads data into dataset class
DATASET = AudioDataset('res/DigiScope Dataset', 'upsampled_training_labels.csv')

# Splits dataset into training and test data with proportions of 0.9 and 0.1 respectively
train, test = random_split(DATASET, (0.9, 0.1))

# Creates train and test loaders using pytorch DataLoader
TRAIN_LOADER = DataLoader(dataset=train, shuffle=True, batch_size=64, drop_last=True)
TEST_LOADER = DataLoader(dataset=test, shuffle=True, batch_size=len(test), drop_last=True)
    

# Trains model on CUDA GPU if available, otherwise trains on CPU
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = ConvolutionalNN().to(DEVICE)
print(f'Running on {DEVICE}')


# Outputs summary of model architecture
summary(model, (5, 128), batch_size=64)


def epoch_function(model, train_dataloader, test_datalaoder, loss_function, optimiser):
    # Iterates through the training dataset to train the model for one epoch
    running_loss = 0
    for index, data in enumerate(train_dataloader):
        inputs, labels = data
        outputs = model(inputs.float()) # Gets outputs for training data input

        loss = loss_function(outputs.float(), labels.float()) # Calculates loss on training data

        optimiser.zero_grad()
        loss.backward() # Performs back propagation to find gradients based on loss
        optimiser.step() # Adjusts model parameters based on back propagated gradients
        
        running_loss += loss.detach()
    mean_training_loss = running_loss / (index+1) # Calculates mean training loss
    
    running_loss = 0
    running_accuracy = 0


    # Iterates through test dataset to test model after every epoch
    for index, test_data in enumerate(test_datalaoder):
        test_inputs, test_labels = test_data
        test_outputs = model(test_inputs.float()) # Gets outputs for test data input

        test_loss = loss_function(test_outputs.float(), test_labels.float()) # Calculates loss on test data

        running_loss += test_loss.detach()

        _, test_labels = torch.max(test_labels, 1)
        _, predictions = torch.max(test_outputs, 1) # Gets model predictions on test data using modle outputs
        correct = torch.sum(predictions == test_labels)
        running_accuracy += correct.detach() / test_labels.size(dim=0)
    
    mean_test_loss = running_loss / (index+1) # Calculates mean test loss
    mean_test_accuracy = running_accuracy / (index+1) # Calculates mean proportion of test data predicted correctly

    return mean_training_loss, mean_test_loss, mean_test_accuracy


def train_model(model, train_dataloader, test_loader, loss_function, optimiser, epochs):
    # Trains model for specified number of epochs and outputs useful metrics
    accuracy_count = 0
    for epoch in tqdm(range(epochs)):
        train_loss, test_loss, test_accuracy = epoch_function(model, train_dataloader, test_loader, loss_function, optimiser)

        print(f'\nEpoch: {epoch + 1}\t\tLoss: {train_loss:.5f}\t\tTest set loss: {test_loss:.5f}\t\tAccuracy: {test_accuracy:.5f}')

        # Stop training when mean testing accuracy is regularly over 98%
        if test_accuracy >= 0.98:
            accuracy_count += 1
            if accuracy_count >= 5:
                print('----------------------------------------------------------------\nAccuracy is above 98%. Stopping train.')
                break
    
    print('----------------------------------------------------------------')
    print('Training complete!')
    print(f'\nFinal:\t\tLoss: {train_loss:.5f}\t\tTest set loss: {test_loss:.5f}\t\tAccuracy: {test_accuracy:.5f}')


loss_function = nn.CrossEntropyLoss() # Defines loss function
optimiser = torch.optim.AdamW(model.parameters(), weight_decay=1e-3) # Defines optimiser with weight decay


# Trains model for a maximum of 300 epochs
train_model(model, TRAIN_LOADER, TEST_LOADER, loss_function, optimiser, 3000)

# Saves model parameters to model.pth
torch.save(model.state_dict(), 'testing/model.pth')
