import gym
import gym.envs.toy_text.frozen_lake as fl
import numpy as np
import matplotlib.pyplot as plt

# environment
env = gym.make('FrozenLake-v0')#, is_slippery=False)
env.reset()

QFunction = np.zeros((env.nS, env.nA))
alpha = 0.01
gamma = 0.9
epsilon = 0.2

state = env.s
reward = 0
done = False
info = {}

goal_found = 0
accuracy = []


# Gets the next action to take given the current state
# Uses gama to decide when to randomly select an action (Explore) and when to select based on Q value (Exploit)
def get_action(s, best = False):
    if best or np.random.random() < epsilon:
        return np.argmax(QFunction[s])
    return np.random.randint(0, env.action_space.n)


def simulate(training = True):
    global state, reward, done, info, goal_found
    while True:
        env.render()

        # stores current state so it can be used to calculate update to Q function
        old_state = state

        # do some action, if we are testing always use q-function to get best move
        action = get_action(old_state, best=not training)
        state, reward, done, info = env.step(action)
        if training:
            # Update Q value
            # i think there is something wrong with my update function
            QFunction[old_state, action] = (1 - alpha)*QFunction[old_state, action] + alpha * (reward + gamma * QFunction[state, get_action(state, best=True)])

        # stop condition
        if done:
            if reward == 1:
                goal_found += 1
            break



    # reset variables run game again
    env.reset()
    state = env.s
    reward = 0
    done = False
    info = {}


# Training
for i in range(10000):
    simulate()
    if i % 100 == 0:
        # test
        goal_found = 0
        for j in range(100):
            simulate(training=False)
        accuracy.append([i/100, goal_found/100])

accuracy = np.matrix(accuracy)
plt.plot(accuracy[:, 0], accuracy[:, 1])
plt.show()

env.close()

# Jointly optimized:
# chess where one action determines which piece to move and the other action where to move that piece
#

