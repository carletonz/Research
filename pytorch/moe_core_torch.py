# this is based off of:
# https://royf.org/pub/pdf/Yan2019Multi.pdf
# https://github.com/AndyLc/mtl-multi-clustering

import torch
#import torchvision
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

M = 4 # Number of experts
N = 2 # Number of tasks
CONNECTION_SIZE = 128 # output size of expert and input size of task head
GE_FUNCTION = "sf" # gradient estimator to use: "sf" = score function, "mv" = measure-valued


# From: https://github.com/jmichaux/dqn-pytorch
class Expert_CNN(nn.Module):
    def __init__(self, in_channels=4, n_actions=4):
        """
        Initialize Deep Q Network

        Args:
            in_channels (int): number of input channels
            n_actions (int): number of outputs
        """
        super(Expert_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.head = nn.Linear(512, CONNECTION_SIZE)

    def forward(self, x):
        x = x.float() / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))

        return self.head(x)

# expert where inputs are color images
# (f)
class Expert_conv(nn.Module):
    def __init__(self):
        super(Expert_conv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=5) # padding 0
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5) # padding 0
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(16*22*22, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, CONNECTION_SIZE)
    
    def forward(self, x):
        conv1_output = self.pool(F.relu(self.conv1(x)))
        conv2_output = self.pool(F.relu(self.conv2(conv1_output)))
        
        reshape_output = conv1_output.view(-1, 16*22*22)
        fc1_output = F.relu(self.fc1(reshape_output))
        fc2_output = F.relu(self.fc2(fc1_output))
        y = self.fc3(fc2_output)
        
        return y

    def output_size(self, x, y):
        x = int(x)
        y = int(y)
        return int((x-5)/2) + ((x-5)%2), int((y-5)/2) + ((y-5)%2)

# experts where inputs are vectors
# (f)
class Expert_linear(nn.Module):
    def __init__(self, input_size):
        super(Expert_linear, self).__init__()
        self.hl1 = nn.Linear(input_size, 128)
        #self.hl2 = nn.Linear(256, 256)
        #self.hl3 = nn.Linear(256, 256)
        #self.hl4 = nn.Linear(256, 256)
        self.hl5 = nn.Linear(128, CONNECTION_SIZE)
    
    def forward(self, x):
        hl1_output = F.relu(self.hl1(x))
        #hl2_output = F.relu(self.hl2(hl1_output))
        #hl3_output = F.relu(self.hl3(hl2_output))
        #hl4_output = F.relu(self.hl4(hl3_output))
        y = self.hl5(hl1_output)
        return y

# Decide to use a specific expert on a task
# Weights decide how much each expert matters
# (b / w)
class Gating(nn.Module):
    def __init__(self):
        super(Gating, self).__init__()
        #self.weights = nn.Parameter(torch.zeros([M, N]))
        self.prob_history = []
        self.logits = nn.Parameter(torch.zeros([M, N]))

        self.prob = torch.zeros([M,N])
        self.save_index = 0

        #self.mapping = torch.eye(N)
        #self.mapping = torch.tensor([[1.0],]).to(device)
    
    def forward(self, x, extra_loss):
        """
        :param x: outputs from experts B x M x F
        :param extra_loss
        :return B x N x F : 
        """
        bernoulli = torch.distributions.bernoulli.Bernoulli(logits = self.logits)#probs=self.mapping)

        b = bernoulli.sample(torch.Size([x.shape[0]]))
        w = b#self.weights * b

        # depeds on b
        self.prob = bernoulli.probs
        #self.save_stats()

        # should be a funcition for log probs
        logits_loss = self.get_logits_loss(bernoulli, b)
        
        #outer product, sum
        output = torch.einsum('bmn, bmf->bnf', w, x)# need to be all the same type

        return output, extra_loss + logits_loss

    def get_logits_loss(self, distribution, b):
        
        if GE_FUNCTION == "sf":
            return self._logits_loss_sf(distribution, b)
        elif GE_FUNCTION == "mv":
            return self._logits_loss_mv(distribution, b)

    def _logits_loss_sf(self, distribution, b):
        return torch.sum(distribution.log_prob(b), 1)

    def _logits_loss_mv(self, distribution, b):
        pass
        #uniform = torch.distributions.unifrom.Uniform(torch.tensor([0]), torch.tensor([1]))
        #b_uniform = (uniform.sample(b.shape) > .5).to(torch.float)
        #loss = (2**(k))*((-1)**(1-b))*(1-distribution.log_prob(b))*distribution.log_prob(b_uniform))

    def save_stats(self):
        if self.save_index % 10 != 0:
            self.save_index += 1
            return
        
        #if not os.path.isdir(output_dir+"/weights"):
        #    os.makedirs(output_dir+"/weights")
        #np.save(output_dir+"/weights/weights"+str(self.save_index), self.weights.detach().cpu().numpy())
        self.prob_hist.append(self.prob.detach().cpu().numpy())
        self.save_index += 1
    def save_prob_history(self, output_dir):
        temp = np.stack(self.prob_hist, axis=0)
        np.save(output_dir+"probs", temp)


# Task Head for each task
# 'a' is the size of the output space for each task
# (g)
class Task(nn.Module):
    def __init__(self, task_output_size):
        super(Task, self).__init__()
        self.hl1 = nn.Linear(CONNECTION_SIZE, task_output_size)
        #self.hl2 = nn.Linear(128, 64)
        #self.hl3 = nn.Linear(64, task_output_size)
    
    def forward(self, x):
        #hl1_output = F.relu(self.hl1(x))
        #hl2_output = F.relu(self.hl2(hl1_output))
        y = self.hl1(x)
        return y

# (m)
class MixtureOfExperts(nn.Module):
    def __init__(self, input_size, task_output_size, input_type="I"):
        super(MixtureOfExperts, self).__init__()
        self.experts = None
        self.expert_type = None
        if input_type == "I": # image
            self.experts = nn.ModuleList([
                Expert_CNN() for i in range(M)
            ])
            self.expert_type = 0
        elif input_type == "V": # vector
            self.experts = nn.ModuleList([
                Expert_linear(input_size) for i in range(M)
            ])
            self.expert_type = 1
        else:
            raise ValueError()
        
        self.gates = Gating()
        N = len(task_output_size)
        self.taskHeads = nn.ModuleList([
                Task(task_output_size[i]) for i in range(N)
            ])
        
        self.train = True
    
    def forward(self, x):
        """
        :param x: tensor containing a batch of images / states / observations / etc.
        """
        batched = True
        # makes sure experts are batched and have correct number of input dimentions
        if self.expert_type == 0 and len(x.shape) > 4: # Images
            raise ValueError("Expecting batched images, got {}".format(x.shape))
        if self.expert_type == 1:
            if len(x.shape) == 1:
                x = x[None]
                batched = False
            if len(x.shape) > 2: # Vectors
                raise ValueError("Expecting batched vectors, got {}".format(x.shape))

        B = x.shape[0] # batch size
        
        # in: B x width x height
        # out: B x M x F
        expert_output = torch.stack([self.experts[i](x) for i in range(M)], dim = 1)
        
        # in: B x M x F
        # out: B x N x F
        self.cumulative_logits_loss = torch.zeros(B, N).to(device)
        gates_output, self.cumulative_logits_loss = self.gates(expert_output, self.cumulative_logits_loss)
        
        # in: B x N x F
        # out: B x O
        # O is the sum of all task output sizes
        # cancatinate this so all task outputs are in the same dimention
        taskHead_output = [self.taskHeads[i](gates_output[:,i]) for i in range(N)]

        return taskHead_output, batched, expert_output.detach().cpu().numpy()
    
    def get_loss(self, loss):
        if GE_FUNCTION == "sf":
            return self._get_loss_sf(loss)
        elif GE_FUNCTION == "mv":
            return self._get_loss_mv(loss)
    
    def _get_loss_sf(self, loss):# score function gradient estimator
        #return loss.mean()
        logits_loss = self.cumulative_logits_loss*loss.detach()
        total_loss = logits_loss + loss
        return total_loss.mean()

    def _get_loss_mv(self, loss):# measure-valued gradient estimator
        pass

if __name__ == "__main__":
    moe = MixtureOfExperts(10, [4,5])
    i = torch.randn(10, dtype=torch.float)
    print(moe(i))


