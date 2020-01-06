import numpy as np
import gym

env = gym.make('FrozenLake-v0', is_slippery=False)
V = np.zeros(env.nS)
gamma = 0.9

for i in range(200):
    for s in range(env.nS):
        temp = []
        for a in range(env.nA):
            temp.append(sum([prob * (reward + gamma * V[next_state]) for prob, next_state, reward, done in env.P[s][a]]))
        temp = np.array(temp)
        V[s] = np.max(temp)
print(V)
