import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
#import pdb; pdb.set_trace()
# Define a simple convolutional neural network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 32, 128)
        self.fc2 = nn.Linear(128, 10)  # Deliberate error: Should have 64 * 7 * 7 output features

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
#breakpoint()
# Create an instance of the model
model = CNN()
# Define some dummy input data
input_data = torch.randn(1, 1, 28, 28)
# Forward pass through the model
output = model(input_data)
