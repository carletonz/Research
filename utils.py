from gym import spaces
import gym
import numpy as np
import torch

# set of environments
class EnvSet(gym.Env):
    def __init__(self, env_fns):
        self.envs = [env_fns[i]() for i in range(len(env_fns))]

        action_space_low = np.concatenate([e.action_space.low for e in self.envs])
        action_space_high = np.concatenate([e.action_space.high for e in self.envs])
        self.action_sizes = list(np.concatenate([e.action_space.shape for e in self.envs]))
        
        self.action_splits = [sum(self.action_sizes[:i+1])for i in range(len(self.action_sizes))]
        
        obs_space_low = np.concatenate([e.observation_space.low for e in self.envs])
        obs_space_high = np.concatenate([e.observation_space.high for e in self.envs])
        self.obs_splits = np.concatenate([e.observation_space.shape for e in self.envs]).sum()
        
        self.action_space = spaces.Box(action_space_low, action_space_high)
        self.observation_space = spaces.Box(obs_space_low, obs_space_high)
        
        self.reward_weights = np.array([1.0, 0.0])

    def step(self, action):
        # List of actions separated by environment
        a = np.split(action, self.action_splits)
        #List of observation, reward, done, info tuple for each environment
        env_state = [self.envs[i].step(a[i]) for i in range(len(self.envs))]

        obs = np.concatenate([env_state[i][0] for i in range(len(env_state))])
        reward = np.array([env_state[i][1]*self.reward_weights[i] for i in range(len(env_state))]).sum()
        done = np.all([env_state[i][2] for i in range(len(env_state))])
        info = np.array([env_state[i][1]*self.reward_weights[i] for i in range(len(env_state))])
        
        return obs, reward, done, info
    
    def reset(self):
        return np.concatenate([env.reset() for env in self.envs])

    def render(self):
        for env in self.envs:
            env.render()
    
    def close(self):
        pass
    
    def seed(self):
        pass

    def __len__(self):
        return len(self.envs)
