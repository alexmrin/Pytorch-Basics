import torch
import torch.nn as nn
import numpy as np

#1 model

class mnist_convnet(nn.Module):
    def __init__(self, num_filters, filter_size):
        super().__init__()
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, self.num_filters, self.filter_size)
        self.maxPooling = nn.MaxPool2d(2)
        self.linear = nn.Linear(self.num_filters * ((28 - self.filter_size + 1)//2)**2, 10)

    def forward(self, x):
        # turn x into shape (batch_size, 1, 28, 28)
        x = x.view(-1, 1, 28, 28)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.maxPooling(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def evaluate(self, data):
        correct = 0
        for input, label in zip(data[0], data[1]):
            y_predicted = self.forward(input)
            y_predicted = self.softmax(y_predicted)
            y_predicted = torch.argmax(y_predicted)

            # compare with label
            if y_predicted.item() == label.item():
                correct += 1

        return correct, data[0].shape[0]

    def predict_input(self, x):
        y_predicted = self.forward(x)
        y_predicted = self.softmax(y_predicted)
        y_predicted = torch.argmax(y_predicted).item()
        return y_predicted