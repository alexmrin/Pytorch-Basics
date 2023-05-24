import mnist_loader
import mnist_net
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
import math
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
import mnist_convnet as cnn

device = torch.device("mps")

# preprocess data

class MNIST_dataset(Dataset):
    def __init__(self):
        self.training_data, self.validation_data, self.test_data = mnist_loader.load_data()
        self.transform = ToTensor()
        self.training_data = self.transform(self.training_data)
        self.validation_data = self.transform(self.validation_data)
        self.test_data = self.transform(self.test_data)
    
    def __len__(self):
        return self.training_data[0].shape[0]
    
    def __getitem__(self, index):
        return self.training_data[0][index], self.training_data[1][index]
    
    def get_test_data(self):
        return self.test_data
    
    def get_validation_data(self):
        return self.validation_data
    
    def get_training_data(self):
        return self.training_data
    
class ToTensor:
    def __call__(self, data):
        data = torch.from_numpy(data[0]).float(), torch.from_numpy(data[1]).long()
        return data
    
        
model = cnn.mnist_convnet(8, 5)
# model.to(device)

batch_size = 50
mnist_data = MNIST_dataset()
dataloader = DataLoader(dataset=mnist_data, batch_size=batch_size, shuffle=True)
# test and validation data
test_data = mnist_data.get_test_data()
validation_data = mnist_data.get_validation_data()
training_data = mnist_data.get_training_data()

# loss and optimizer
learning_rate = 0.05
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# lmbda is the constant we use for each weight term
lmbda_reg = 10

# training loop
num_epochs = 50
n_iterations = math.ceil(len(mnist_data)/batch_size)

# scheduler
step_size = num_epochs/3
gamma = 0.1
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

epochs = [i for i in range(num_epochs)]
losses = []
accuracies = []

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # inputs = inputs.to(device=device)
        # labels = labels.to(device=device)
        # forward backward pass
        y_predicted = model(inputs)
        l2_regularization = torch.tensor(0.0)
        for name, param in model.named_parameters():
            if 'weight' in name:
                ## each weight is a weight matrix for each layer and we must add each invidual weight squared
                l2_regularization += torch.sum(param**2)
        loss = criterion(y_predicted, labels) + l2_regularization * lmbda_reg/len(mnist_data)

        loss.backward()

        # update step
        optimizer.step()
        scheduler.step()
        # empty gradients
        optimizer.zero_grad()

    # scheduler.step()
    correct, total = model.evaluate(validation_data)
    losses.append(loss.item())
    accuracies.append(correct/total)

    print(f"epoch: {epoch+1}/{num_epochs}, loss: {loss}, performance: {correct}/{total}, regularization term: {l2_regularization}")

correct, total = model.evaluate(test_data)
print(f"The network got {correct/total * 100:.2f}% on the test data.")

torch.save(model.state_dict(), "./mnist_feedforward.pth")

plt.figure(1)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(epochs, losses, 'ro')
plt.title("loss function")

plt.figure(2)
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.plot(epochs, accuracies, 'b')
plt.title("performance")
plt.show()
