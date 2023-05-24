import torch
import torch.nn as nn
import numpy as np
device = torch.device("mps")

# 1) model
class mnistNet(nn.Module):
    # layers will be a list of length 3 that includes the number of neurons in each layer
    def __init__(self, layers):
        super().__init__()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.linear1 = nn.Linear(layers[0], layers[1])
        self.linear2 = nn.Linear(layers[1], layers[2])
        self.linear3 = nn.Linear(layers[2], layers[3])

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
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
    
class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)