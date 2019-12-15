import gym
import gym.envs.toy_text.frozen_lake as fl
import numpy as np
import matplotlib.pyplot as plt

# Show that a joint optimization can be efficiently trained
# Use frozen lake where one action is to go horizontally or vertically
# Other action is to go in the positive (up, right) or negative (down, left) direction

# environment
# si_slippery=True --> 33% chance chosen action is taken, Default
# si_slippery=False --> 100% chance chosen action is taken
env = gym.make('FrozenLake-v0', is_slippery=False)
env.reset()

agent1_QFunction = np.zeros((env.nS, int(env.nA/2))) # horizontal or vertical
agent2_QFunction = np.zeros((env.nS, int(env.nA/2))) # positive or negative
alpha = 0.007
gamma = 0.9
epsilon = 0.02
episodes = 30000

state = env.s
reward = 0
done = False
info = {}

goal_found = 0
a1_sum_q = 0
a2_sum_q = 0
accuracy = []
q_values = []


# Gets the next action to take given the current state
# Uses gama to decide when to randomly select an action (Explore) and when to select based on Q value (Exploit)
def get_action_1(s, best = False):
    if best or np.random.random() < epsilon:
        return np.argmax(agent1_QFunction[s])
    return np.random.randint(0, env.nA/2)


# Gets the next action to take given the current state
# Uses gama to decide when to randomly select an action (Explore) and when to select based on Q value (Exploit)
def get_action_2(s, best = False):
    r = np.random.random()
    if best or r < epsilon:
        return np.argmax(agent2_QFunction[s])
    return np.random.randint(0, env.nA / 2)


def simulate(training = True):
    global state, reward, done, info, goal_found, a1_sum_q, a2_sum_q
    while True:
        env.render()

        # stores current state so it can be used to calculate update to Q function
        old_state = state

        # do some action, if we are testing always use q-function to get best move
        action_1 = get_action_1(old_state, best=not training)
        action_2 = get_action_2(old_state, best=not training)
        state, reward, done, info = env.step(2*action_2+action_1)

        a1_sum_q += agent1_QFunction[old_state, action_1]
        a2_sum_q += agent2_QFunction[old_state, action_2]

        if training:
            # Update Q values
            agent1_QFunction[old_state, action_1] = (1 - alpha) * agent1_QFunction[old_state, action_1] + alpha * (reward + gamma * agent1_QFunction[state, get_action_1(state, best=True)])
            agent2_QFunction[old_state, action_2] = (1 - alpha) * agent2_QFunction[old_state, action_2] + alpha * (reward + gamma * agent2_QFunction[state, get_action_2(state, best=True)])

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
continueTraining = True
for i in range(episodes):
    simulate()
    if i % 100 == 0:
        # test
        goal_found = 0
        for j in range(100):
            a1_sum_q = 0
            a2_sum_q = 0
            simulate(training=False)
        q_values.append([a1_sum_q, a2_sum_q])
        accuracy.append([i/100, goal_found/100])
        if accuracy[-1][1] > .7:
            continueTraining = False

accuracy = np.matrix(accuracy)
plt.plot(accuracy[:, 0], accuracy[:, 1])
plt.xlabel('x100 Updates')
plt.ylabel('Accuracy')
plt.show()

q_values = np.matrix(q_values)
plt.plot(np.arange(q_values.shape[0]), q_values[:, 0])
plt.xlabel('x100 updates')
plt.ylabel('Total Reward 1')
plt.show()

plt.plot(np.arange(q_values.shape[0]), q_values[:, 1])
plt.xlabel('x100 updates')
plt.ylabel('Total Reward 2')
plt.show()

print(q_values)

env.close()

# Separately optimized
# frozen lake where one action determines horizontal or vertical movement and the other action determines positive or
# negative direction

