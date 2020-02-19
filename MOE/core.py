# this is based off of:
# https://royf.org/pub/pdf/Yan2019Multi.pdf
# https://github.com/AndyLc/mtl-multi-clustering

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from macros import *

# expert where inputs are color images
# (f)
class Expert_conv(nn.Module):
    def __init__(self):
        super(Expert_conv, self)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5) # padding 0
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5) # padding 0
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, CONNECTION_SIZE)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        x = self.fc3(x)
        
        return x

# experts where inputs are vectors
# (f)
class Expert_linear(nn.Module):
    def __init__(self, input_size):
        super(Expert_linear, self)
        self.fc1 = nn.Linear(input_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, CONNECTION_SIZE)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

# Decide to use a specific expert on a task, output should be a vector of 1s and 0s of length # of tasks
# Weights decide how much each expert matters
# (b / w)
class Gating(nn.Module):
    def __init__(self):
        super(Gating, self)
        self.hl1 = nn.Linear(T, 120)
        self.hl2 = nn.Linear(120, 84)
        self.hl3 = nn.Linear(84, T*M)
        
        self.logits = nn.Parameters(torch.zeros([T, M]))
        self.logits_loss = 0
    
    def forward(self, x):
        """
        :param x: tensor containing a task number
        """
        bernoulli = torch.Bernoulli(logits=self.logits)
        task = x
        
        # one hot encoding
        # task numbers start from 0
        x = torch.zeros([T])
        x[task] = 1
        
        x = F.relu(self.hl1(x))
        x = F.relu(self.hl2(x))
        x = self.hl3(x)
        
        b = bernoulli.sample()
        self.logits_loss = torch.sum(torch.log(bernoulli.probs())[task])
        
        return x.view(T,M) * b
    
    def get_loss(inputs, targets):
        return self.logits_loss * inputs.detach()

# Task Head for each task
# A is the size of the output space for each task
# (g)
class Task(nn.Module):
    def __init__(self, A):
        super(Task, self)
        self.hl1 = nn.Linear(CONNECTION_SIZE, 120)
        self.hl2 = nn.Linear(120, 84)
        self.hl3 = nn.Linear(84, A)
    
    def forward(self, x):
        x = F.relu(self.hl1(x))
        x = F.relu(self.hl2(x))
        x = self.hl3(x)
        return x

class MixtureOfExperts(nn.Module):
    def __init__(self, task_info, input_type="I", input_size=None):
        super(MixtureOfExperts, self)
        self.experts = None
        if input_type == "I":
            self.experts = nn.ModuleList([
                Expert_conv() for i in range(M)
            ])
        elif input_type == "V" and input_size != None:
            self.experts = nn.ModuleList([
                Expert_linear(input_size) for i in range(M)
            ])
        else:
            raise ValueError()
        
        self.gates = Gating()
        self.taskHeads = nn.ModuleList([
                Task(task_info.nA) for i in range(M)
            ])
        
        self.train = True
    
    def forward(self, x, task):
        """
        :param x: tensor containing an image / state / observation / etc.
        :param task: tensor containing a task number
        """
        w = self.gates(task)
        if self.train:
            # pass x to all experts
            x = [self.experts[i](x) * w[task][i] for i in range(M)]
            x = sum(x)
        else:
            # pass x to experts where weights are greater than 0
            temp = []
            for i in range(M):
                if w[task][i] > 0:
                    temp.append(self.experts[i](x) * w[task][i])
            x = sum(temp)
        
        x = self.taskHeads[task](x)
        return x
    
    def set_training(train = True):
        self.train = train
    
    def get_loss(inputs, targets):
        loss = self.gates.get_loss(inputs, targets)
        loss = loss + inputs
        return loss