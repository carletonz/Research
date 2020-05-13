from gym import spaces
import gym
import numpy as np
import torch
from gym.wrappers import Monitor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set of environments
class EnvSet(gym.Env):
    def __init__(self, env_fns, video_path = None):
        self.envs = [env_fns[i]() for i in range(len(env_fns))]
        if video_path is not None:
            self.envs = [(Monitor(self.envs[i], video_path[i]) if video_path[i] is not None else self.envs[i]) for i in range(len(self.envs))]

        action_space_low = np.concatenate([e.action_space.low for e in self.envs])
        action_space_high = np.concatenate([e.action_space.high for e in self.envs])
        self.action_sizes = list(np.concatenate([e.action_space.shape for e in self.envs]))
        
        self.action_splits = [sum(self.action_sizes[:i+1])for i in range(len(self.action_sizes))]
        
        obs_space_low = np.concatenate([e.observation_space.low for e in self.envs])
        obs_space_high = np.concatenate([e.observation_space.high for e in self.envs])
        self.obs_splits = np.concatenate([e.observation_space.shape for e in self.envs]).sum()
        
        self.action_space = spaces.Box(action_space_low, action_space_high)
        self.observation_space = spaces.Box(obs_space_low, obs_space_high)
        
        self.reward_weights = np.array([1.0, 0.75])

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
        for env in self.envs:
            env.close()
    
    def seed(self):
        pass

    def __len__(self):
        return len(self.envs)





class EnvDecomposed(gym.Env):
    def __init__(self, env_fn, action_sizes, remap):
        self.env = env_fn()

        self.action_sizes = action_sizes # list of the size of each decomposed action space
        self.remap = remap # list of indexes of action space

        self.obs_splits = self.env.observation_space.shape[0]

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def step(self, action):
        # List of actions separated by environment
        a = action[self.remap]
        obs, reward, done, _ = self.env.step(a)
        return obs, reward, done, np.ones(len(self.action_sizes))*reward

    def reset(self):
        return self.env.reset()

    def render(self):
        self.env.render()

    def close(self):
        pass

    def seed(self):
        pass

    def __len__(self):
        return len(self.action_sizes)




def run_model(env_fn, model_path, video_path = None, ):
    env = env_fn(video_path)
    model = torch.load(model_path)
    
    obs, done, ep_ret, ind_ret = env.reset(), False, 0, np.zeros(len(env))
    ep_returns = []
    individual_returns = [list() for i in range(len(env))]


    episodes = 5
    steps_per_episode = 1000
    total_steps = episodes * steps_per_episode

    for i in range(total_steps):
        obs, reward, done, info = env.step(model.act(torch.as_tensor(obs, dtype=torch.float32).to(device), True))
        ep_ret += reward
        ind_ret += info

        if (i+1)%steps_per_episode == 0 or done:
            ep_returns.append(ep_ret)
            [individual_returns[i].append(ind_ret[i]) for i in range(len(ind_ret))]
            obs, done, ep_ret, ind_ret = env.reset(), False, 0, np.zeros(len(env))
    
    print("===== Overall =====")
    print(f"Average return: {np.mean(ep_returns)}")
    print(f"Std return    : {np.std(ep_returns)}")
    print(f"Min return    : {np.min(ep_returns)}")
    print(f"Max return    : {np.max(ep_returns)}")
    
    for i in range(len(env)):
        print("===== env{} =====".format(i))
        print(f"Average return: {np.mean(individual_returns[i])}")
        print(f"Std return    : {np.std(individual_returns[i])}")
        print(f"Min return    : {np.min(individual_returns[i])}")
        print(f"Max return    : {np.max(individual_returns[i])}")
    env.close()
