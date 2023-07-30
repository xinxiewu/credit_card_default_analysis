"""
deep_learning.py contains neural networks:
    1. MyNetwork
    2. Training process
"""
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import torch
import os

class MyNetwork(nn.Module):
    def __init__(self, input_feature, dim1, dim2, drop, output_feature):
        super(MyNetwork, self).__init__()
        self.model1 = nn.Sequential(
            nn.Linear(input_feature, dim1, bias=True),
            nn.ReLU()
        )
        self.model2 = nn.Sequential(
            nn.Linear(dim1, dim2, bias=True),
            nn.Dropout(drop),
            nn.ReLU()
        )
        self.model3 = nn.Sequential(
            nn.Linear(dim2, output_feature, bias=True),
        )   
    
    def forward(self, input):
        x1 = self.model1(input)
        x2 = self.model2(x1)
        output = self.model3(x2)
        return output

def training_nn(epochs=500, learning_rate=0.01, loss_func='CE', optimizer_para='SGD',
                input_feature=23, output_feature=2, dim1=36, dim2=108, dropout=0.8, val_size = None,
                train_loader=None, val_loader=None, test_loader=None, device=None, save=False, output=None):
    # Initialize model
    mynetwork = MyNetwork(input_feature=input_feature, dim1=dim1, dim2=dim2, drop=dropout, output_feature=output_feature)
    mynetwork.to(device)

    # Define loss function & optimizer
    if loss_func == 'CE':
        loss_fn = nn.CrossEntropyLoss()
        loss_fn.to(device)
    if optimizer_para == 'SGD':
        optimizer = optim.SGD(mynetwork.parameters(), lr=learning_rate)

    # Epoches, training & validation
    total_training_step, total_val_step = 0, 0
    writer = SummaryWriter(output)
    for i in range(epochs):
        if (i+1)%100 == 0:
            print(f"------------epoch: {i+1}------------")
        """
        Training
        """
        mynetwork.train()
        for data in train_loader:
            inputs, labels = data
            inputs = inputs.view(-1, input_feature)
            labels = labels.squeeze().long()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = mynetwork(inputs)
            loss = loss_fn(outputs, labels.long())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_training_step += 1
            if total_training_step % 100 == 0:
                print(f"Traning: {total_training_step}; Loss: {loss.item()}")
                writer.add_scalar("train_loss", loss.item(), total_training_step)

        """
        Validation
        """
        mynetwork.eval()
        total_test_loss, total_acc = 0, 0
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                inputs = inputs.view(-1, input_feature)
                labels = labels.squeeze().long()
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = mynetwork(inputs)
                loss = loss_fn(outputs, labels)
                total_test_loss += loss.item()
                acc = (outputs.argmax(1) == labels).sum()
                total_acc += acc
        print(f"total test loss: {total_test_loss}")
        print(f"total accuracy: {total_acc/val_size}")
        writer.add_scalar("test_loss", total_test_loss, total_val_step)
        writer.add_scalar("test_accuracy", total_acc/val_size, total_val_step)
        total_val_step += 1
        if save == True:
            torch.save(mynetwork, f"mynetwork_{i}.pth")

    writer.close()