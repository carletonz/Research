import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from pytorch import moe_core_torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class MOEConnector(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit, env):
        super().__init__()
        print(env.obs_splits, env.action_sizes)
        self.net = moe_core_torch.MixtureOfExperts(env.obs_splits, env.action_sizes)
        self.pis = nn.ModuleList([
            SquashedGaussianMLPActor(env.obs_sizes[i], env.action_sizes[i], [env.action_sizes[i]], activation, act_limit) for i in range(len(env))
            ])

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out, batched = self.net(obs.to(device))
        actor_out = [self.pis[i](net_out[i], deterministic, with_logprob) for i in range(len(self.pis))]
        
        #actor_out[1][0].detach_()
        action = torch.cat([actor_out[i][0] for i in range(len(self.pis))], dim=1)
        if with_logprob:
            #actor_out[1][1].detach_()
            logp_pi = torch.stack([actor_out[i][1] for i in range(len(self.pis))], dim=1)
        else:
            logp_pi = None

        if not batched:
            action = action[0]
            if logp_pi != None:
                print(logp_pi)
                raise "error"
        return action, logp_pi


LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        #if env != None:
        #    self.net = moe_core_torch.MixtureOfExperts(env.obs_splits, env.action_sizes)
        #else:
        #    self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = obs.to(device)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
            #logp_pi = logp_pi.split(self.act_sizes, dim=1)
            #logp_pi = torch.stack([x.sum(dim=-1) for x in logp_pi], dim=1)

        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [moe_core_torch.N], activation)


    def forward(self, obs, act):
        obs = obs.to(device)
        act = act.to(device)
        q = self.q(torch.cat([obs, act], dim=-1))
        return q#torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU, env=None):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]
        
        # build policy and value functions
        self.pi = MOEConnector(obs_dim, act_dim, hidden_sizes, activation, act_limit, env)
        #SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit, env)

        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        obs = obs.to(device)
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.cpu().numpy()
