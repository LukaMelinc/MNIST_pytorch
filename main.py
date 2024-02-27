import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import time
import matplotlib.pyplot as plt

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)

'''for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(mnist_trainset[i][0], cmap= 'gray')
plt.show()
'''


# Device configuration
device = torch.device('cpu') # training on the cpu

# Hyper-parameters 
input_size = 784 # 28x28
hidden_size = 500 
num_classes = 10
num_epochs = 10
batch_size = 64
learning_rate = 0.001

# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

examples = iter(test_loader)
example_data, example_targets = next(examples)

print(example_data.shape, example_targets.shape)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out
    
model = NeuralNet(input_size, hidden_size, num_classes)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#training loop
n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        #forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        #backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'epoch {epoch+1}/{num_epochs}, step = {i+1}/{n_total_steps}, loss = {loss.item():.3f}')

def test():

# Izračun deleža pravilno razvrščenih testnih vzorev
    pravilni = 0
    for i in range(len(test_dataset)):
        if test_dataset.label[i] == NeuralNet(test_dataset).label[i]:
            pravilni += 1
        
    natancnost = pravilni / len(test_dataset)
    print(natancnost)
    return 0

test(model)
        
