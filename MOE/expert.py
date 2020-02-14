import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# expert where inputs are color images
class Expert_conv(nn.Module):
    def __init__(self):
        super(Expert_conv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5) # padding 0
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5) # padding 0
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        x = self.fc3(x)
        
        return x

# experts where inputs are vectors
class Expert_linear(nn.Module):
    def __init__(self, input_size):
        super(Expert_linear, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

# Decide to use a specific expert on a task, output should be a vector of 1s and 0s of length # of tasks
class Gating(nn.Module):
    def __init__(self):
        super(Gating, self).__init__()
    
    def forward(self, x):
        return x

# Weights decide how much each expert matters
# T=Number of tasks
class Weights(nn.Module):
    def __init__(self, T):
        self(Weights, self).__init__()
        self.fc1 = nn.Linear(1, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, T)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

# M = number of experts
# A = output space for this specific task
class Task(nn.Module):
    def __init__(self, M, A):
        self(Weights, self).__init__()
        self.fc1 = nn.Linear(1, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, )
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x