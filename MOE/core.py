# this is based off of:
# https://royf.org/pub/pdf/Yan2019Multi.pdf
# https://github.com/AndyLc/mtl-multi-clustering

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

M = 10 # Number of experts
N = 100 # Number of tasks
CONNECTION_SIZE = 10 # output size of expert and input size of task head

# expert where inputs are color images
# (f)
class Expert_conv(nn.Module):
    def __init__(self):
        super(Expert_conv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5) # padding 0
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5) # padding 0
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, CONNECTION_SIZE)
    
    def forward(self, x):
        conv1_output = self.pool(F.relu(self.conv1(x)))
        conv2_output = self.pool(F.relu(self.conv2(conv1_output)))
        
        reshape_output = conv1_output.view(-1, 16*5*5)
        fc1_output = F.relu(self.fc1(reshape_output))
        fc2_output = F.relu(self.fc2(fc1_output))
        y = self.fc3(fc2_output)
        
        return y

# experts where inputs are vectors
# (f)
class Expert_linear(nn.Module):
    def __init__(self, input_size):
        super(Expert_linear, self).__init__()
        self.hl1 = nn.Linear(input_size, 120)
        self.hl2 = nn.Linear(120, 84)
        self.hl3 = nn.Linear(84, CONNECTION_SIZE)
    
    def forward(self, x):
        hl1_output = F.relu(self.hl1(x))
        hl2_output = F.relu(self.hl2(hl1_output))
        y = self.hl3(hl2_output)
        return y

# Decide to use a specific expert on a task
# Weights decide how much each expert matters
# (b / w)
class Gating(nn.Module):
    def __init__(self):
        super(Gating, self).__init__()
        self.weights = nn.Parameter(torch.zeros([M, N]))
        self.logits = nn.Parameter(torch.zeros([M, N]))
    
    def forward(self, x, extra_loss):
        """
        :param x: outputs from experts M x F
        :param task
        :return M x N x B x F
        """
        bernoulli = torch.Bernoulli(logits=self.logits)
        
        b = bernoulli.sample()
        w = self.weights * b
        logits_loss = torch.sum(torch.log(bernoulli.probs()), 0)
        
        #outer product
        output = torch.einsum('bp, bqr->bpqr', w, x)
        
        return output, extra_loss + logits_loss

# Task Head for each task
# 'a' is the size of the output space for each task
# (g)
class Task(nn.Module):
    def __init__(self, task_output_size):
        super(Task, self).__init__()
        self.hl1 = nn.Linear(CONNECTION_SIZE, 120)
        self.hl2 = nn.Linear(120, 84)
        self.hl3 = nn.Linear(84, task_output_size)
    
    def forward(self, x):
        hl1_output = F.relu(self.hl1(x))
        hl2_output = F.relu(self.hl2(hl1_output))
        y = self.hl3(hl2_output)
        return y

# (m)
class MixtureOfExperts(nn.Module):
    def __init__(self, task_output_size, input_type="I", input_size=None):
        super(MixtureOfExperts, self).__init__()
        self.experts = None
        self.expert_type = None
        if input_type == "I":
            self.experts = nn.ModuleList([
                Expert_conv() for i in range(M)
            ])
            self.expert_type = 0
        elif input_type == "V" and input_size != None:
            self.experts = nn.ModuleList([
                Expert_linear(input_size) for i in range(M)
            ])
            self.expert_type = 1
        else:
            raise ValueError()
        
        self.gates = Gating()
        self.taskHeads = nn.ModuleList([
                Task(task_output_size) for i in range(M)
            ])
        
        self.train = True
    
    def forward(self, x, task):
        """
        :param x: tensor containing a batch of images / states / observations / etc.
        :param task: tensor containing a task number
        """
        # makes sure experts are batched and have correct number of input dimentions
        if self.expert_type == 0 and len(x.shape) != 3: # Images
            raise ValueError()
        if self.expert_type == 1 and len(x.shape) != 2: # Vectors
            raise ValueError()
        
        B = x.shape[0] # batch size
        
        # in: B x width x height
        # out: M x B x F
        expert_output = torch.stack([self.experts[i](x) for i in range(M)])
        
        # in: M x B x F
        # out: N x B x F
        self.cumulative_logits_loss = torch.zeros(N)
        gates_output, logits_loss = self.gates(expert_output, logits_loss)
        gates_output = torch.sum(torch.tensor(gates_output), 0)
        
        # in: N x B x F
        # out: N x B x O
        taskHead_output = [self.taskHeads[i](gates_output[i]) for i in range(N)]
        return taskHead_output
    
    def get_loss(loss):
        logits_loss = self.cumulative_logits_loss * loss.detach()
        total_loss = logits_loss + loss
        return total_loss

if __name__ == "__main__":
    task = torch.tensor(0)
    input_type = "I"
    task_output_size = 6
    inputs = None # todo: get actual inputs
    label = None # todo: get actual labels
    
    moe = MixtureOfExperts(task_output_size, input_type)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(moe.parameters(), lr=0.001, momentum=0.9)
    
    output = moe(inputs, task)
    output, action = torch.max(output, 0)
    
    optimizer.zero_grad()
    loss = criterion(output, label)
    loss = moe.get_loss(loss)
    
    loss.backward()
    optimizer.step()
