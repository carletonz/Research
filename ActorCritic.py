import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import moe_core.MixtureOfExperts as moe

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class ConvQFunction(nn.Module):
    # need to implement
    def __init__(self):
        raise NotImplementedError
    
    def forward(self, obs, act):
        raise NotImplementedError

class MLPQFunction(nn.Module):# for vector inputs only

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MOEActorCritic(nn.Module):
    def __init__(self, observation_spaces, action_spaces, input_type="V", hidden_sizes=(256,256), activation=nn.ReLU):
        super().__init__()
        # action spaces and observation spaces all must be continuous
        # currently only works with vector observations and action spaces
        obs_dim = sum([observation_space[i].shape[0] for i in range(len(observation_spaces))])
        act_dim = sum([action_spaces[i].shape[0] for i in range(len(action_spaces))])

        # build policy and value functions
        self.pi = moe(observation_spaces, action_spaces, input_type)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs)
            return a.numpy()