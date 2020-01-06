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

QFunction = np.zeros((env.nS, int(env.nA/2), 2))
alpha = 0.01
gamma = 0.9
epsilon = 0.2
episodes = 50000

state = env.s
reward = 0
done = False
info = {}

goal_found = 0
R = 0
g_t = 1
accuracy = []
R_values = []


# Gets the next action to take given the current state
# Uses gama to decide when to randomly select an action (Explore) and when to select based on Q value (Exploit)
def get_action(s, best = False):
    if best or np.random.random() < epsilon:
        return np.argmax(QFunction[s])
    return np.random.randint(0, env.action_space.n)


def simulate(training = True):
    global state, reward, done, info, goal_found, R, g_t
    while True:
        env.render()

        # stores current state so it can be used to calculate update to Q function
        old_state = state

        # do some action, if we are testing always use q-function to get best move
        action = get_action(old_state, best=not training)
        state, reward, done, info = env.step(action)
        row = int(action / 2)
        col = action % 2

        g_t *= gamma
        R += g_t * reward

        if training:
            # Update Q value
            best_action = get_action(state, best=True)
            b_row = int(best_action/2)
            b_col = best_action%2

            QFunction[old_state, row, col] = (1 - alpha)*QFunction[old_state, row, col] + alpha * (reward + gamma * QFunction[state, b_row, b_col])

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
for i in range(episodes):
    simulate()
    if i % 100 == 0:
        # test
        goal_found = 0
        ave_R = 0
        for j in range(100):
            R = 0
            g_t = 1
            sum_q = 0
            simulate(training=False)
            ave_R += R
        R_values.append(ave_R / 100)
        accuracy.append([i/100, goal_found/100])

accuracy = np.matrix(accuracy)
plt.plot(accuracy[:, 0], accuracy[:, 1])
plt.xlabel('x100 Updates')
plt.ylabel('Accuracy')
plt.show()

R_values = np.array(R_values)
plt.plot(np.arange(R_values.shape[0]), R_values)
plt.xlabel('x100 updates')
plt.ylabel('R')
plt.show()

print(QFunction)

env.close()

# Jointly optimized:
# chess where one action determines which piece to move and the other action where to move that piece
#

