import spinup
import numpy as np
import gym
import time
import torch

#env = "Ant-v2"
env = "HalfCheetah-v2"
class PsudoEnv(gym.Env):
    def __init__(self):
        self.env = gym.make(env)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def reset(self):
        return self.env.reset()

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def seed(self):
        return self.env.seed()

outputDir = "/home/ubuntu/Documents/proj/research/Research/imitation/imit-1603432232"
#spinup.sac_pytorch(
#        PsudoEnv,
#        epochs = 800,
#        logger_kwargs={"output_dir": outputDir})

#check = input("continue? y/n")
#if check == "n":
#    quit()

model_path = outputDir+"/pyt_save/model.pt"

env = PsudoEnv()
model = torch.load(model_path)

obs, done = env.reset(), False

obs_hist = []
act_hist = []

episodes = 10
steps_per_episode = 4000
total_steps = episodes * steps_per_episode

for i in range(total_steps):
    action = model.act(torch.as_tensor(obs, dtype=torch.float32), True)#.to(device)
    obs_hist.append(obs)
    act_hist.append(action)
    obs, reward, done, info = env.step(action)
    
    if (i+1)%steps_per_episode == 0 or done:
        obs, done = env.reset(), False


print(obs_hist)
print(act_hist)
print(outputDir)
np.save(outputDir+"/obshc", np.stack(obs_hist))
np.save(outputDir+"/acthc", np.stack(act_hist))
