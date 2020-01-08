import numpy as np
import gym

env = gym.make('MountainCar-v0')
pos, v = env.reset()
# individualCounts = ((env.high - env.low) / np.array([0.1, 0.01])).astype(int)
# individualCounts[0] += 2
# individualCounts[1] += 1
# stateCount = (individualCounts[0]) * (individualCounts[1])
#
# V = np.zeros(stateCount)
# gamma = 1
#
#
# def hash_state(state):
#     state = ((np.array(state) + np.array([1.2, 0.07])) / np.array([0.1, 0.01])).astype(int)
#     return state[1] * individualCounts[0] + state[0]
#
#
# def expand_state(state):
#     zero = state % individualCounts[0]
#     one = int(state / individualCounts[0])
#
#     zero = zero * 0.1
#     one = one * 0.01
#
#     zero = zero - 1.2
#     one = one - 0.07
#
#     return np.array([zero, one])
#
#
# for i in range(100):
#     for s in range(stateCount):
#         temp = []
#         expanded_state = expand_state(s)
#         if expanded_state[0] >= env.goal_position and expanded_state[1] >= env.goal_velocity:
#             continue
#         for a in range(3):
#             env.state = expanded_state
#             next_state, reward, done, _ = env.step(a)
#             temp.append(reward + gamma * V[hash_state(next_state)])
#
#         temp = np.array(temp)
#         V[s] = np.max(temp)
# s1 = hash_state(np.array([-0.6, 0]))
# s2 = hash_state(np.array([-0.5, 0]))
# s3 = hash_state(np.array([-0.4, 0]))
# s4 = hash_state(np.array([0.4, -0.07]))
#
# print(V[[s1, s2, s3, s4]])
s = 0
for i in range(16):
    env.render()
    _, _, done, _ = env.step(2)
    if not done:
        s += 1
for i in range(38):
    env.render()
    _, _, done, _ = env.step(0)
    if not done:
        s += 1
for i in range(50):
    env.render()
    _, _, done, _ = env.step(2)
    if not done:

        s += 1
    else:
        print("done")
print(s)