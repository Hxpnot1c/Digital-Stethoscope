from data_preprocessor import AudioDataset
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch
from torchsummary import summary
from tqdm import tqdm
from model import ConvolutionalNN
# import os
# import time

dataset = AudioDataset('res/DigiScope Dataset', 'upsampled_training_labels.csv')

train, test = random_split(dataset, (0.9, 0.1))

train_loader = DataLoader(dataset=train, shuffle=True, batch_size=64, drop_last=True)
test_loader = DataLoader(dataset=test, shuffle=True, batch_size=len(test), drop_last=True)
    

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = ConvolutionalNN().to(device)
print(f'Running on {device}')

summary(model, (5, 128), batch_size=64)

def epoch_function(model, train_dataloader, test_datalaoder, loss_function, optimiser):
    running_loss = 0
    for index, data in enumerate(train_dataloader):
        inputs, labels = data
        outputs = model(inputs.float())

        loss = loss_function(outputs.float(), labels.float())

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        
        running_loss += loss.detach()
    mean_training_loss = running_loss / (index+1)
    
    running_loss = 0
    running_accuracy = 0
    for index, test_data in enumerate(test_datalaoder):
        test_inputs, test_labels = test_data
        test_outputs = model(test_inputs.float())

        test_loss = loss_function(test_outputs.float(), test_labels.float())

        running_loss += test_loss.detach()

        _, test_labels = torch.max(test_labels, 1)
        _, predictions = torch.max(test_outputs, 1)
        correct = torch.sum(predictions == test_labels)
        running_accuracy += correct.detach() / test_labels.size(dim=0)
    
    mean_test_loss = running_loss / (index+1)
    mean_test_accuracy = running_accuracy / (index+1)

    return mean_training_loss, mean_test_loss, mean_test_accuracy

def train_model(model, train_dataloader, test_loader, loss_function, optimiser, epochs):
    accuracy_count = 0
    for epoch in tqdm(range(epochs)):
        train_loss, test_loss, test_accuracy = epoch_function(model, train_dataloader, test_loader, loss_function, optimiser)
        print(f'\nEpoch: {epoch + 1}\t\tLoss: {train_loss:.5f}\t\tTest set loss: {test_loss:.5f}\t\tAccuracy: {test_accuracy:.5f}')
        if test_accuracy >= 0.98:
            accuracy_count += 1
            if accuracy_count >= 1:
                print('----------------------------------------------------------------\nAccuracy is above 98%. Stopping train.')
                break
    
    print('----------------------------------------------------------------')
    print('Training complete!')
    print(f'\nFinal:\t\tLoss: {train_loss:.5f}\t\tTest set loss: {test_loss:.5f}\t\tAccuracy: {test_accuracy:.5f}')

loss_function = nn.CrossEntropyLoss()
optimiser = torch.optim.AdamW(model.parameters(), weight_decay=1e-3)

train_model(model, train_loader, test_loader, loss_function, optimiser, 3000)
torch.save(model.state_dict(), 'testing/model.pth')
# time.sleep(10)
# os.system("shutdown /s /t 0")