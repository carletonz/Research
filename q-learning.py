import gym
import numpy as np

# environment
env = gym.make('FrozenLake-v0')
env.reset()

QFunction = np.zeros((16, 4))
alpha = 0.05
gama = 0.9

state = 0
reward = 0
done = False
info = {}

goal_found = 0

# Gets the next action to take given the current state
# Uses gama to decide when to randomly select an action (Explore) and when to select based on Q value (Exploit)
def get_action(obs, best = False):
    if best or np.random.random() < gama:
        # randomly chooses action to do if Q values are all the same
        return np.random.choice(np.flatnonzero(QFunction[obs] == np.max(QFunction[obs])))
    return np.random.randint(0, 4)


for i in range(7000):
    while True:
        env.render()

        # stores current observation so it can be used to calculate update to Q function
        oldObservation = state

        # do some action
        action = get_action(oldObservation)
        state, reward, done, info = env.step(action)

        # Update Q value
        QFunction[oldObservation, action] = (1-alpha)*QFunction[oldObservation, action] + alpha * (reward + gama * QFunction[state, get_action(state, best=True)])

        # stop condition
        if done:
            if reward == 1:
                goal_found += 1
            QFunction[state, 0] = reward
            QFunction[state, 1] = reward
            QFunction[state, 2] = reward
            QFunction[state, 3] = reward
            env.reset()
            break

    # reset variables run game again
    observation = 0
    reward = 0
    done = False
    info = {}

print(goal_found)
print("Left down right up")
print(QFunction)

env.close()
