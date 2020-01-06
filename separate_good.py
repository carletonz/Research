import gym
import numpy as np
import matplotlib.pyplot as plt


class QLearning:
    def __init__(self, environment):

        # position: -1.2 to 0.6
        # speed: -0.07 to 0.07

        # make continuous states as discrete by dividing by 0.1 and 0.01
        # this makes 19 discrete states for position and 15 discrete states for speed
        # with a total of 285 states

        self.individualCounts = ((environment.high - environment.low) / np.array([0.1, 0.01])).astype(int)
        self.individualCounts[0] += 2
        self.individualCounts[1] += 1
        self.stateCount = (self.individualCounts[0]) * (self.individualCounts[1])
        self.actionCount = environment.action_space.n

        self.QFunction = np.zeros((self.stateCount, self.actionCount))
        self.alpha = 0.01
        self.gamma = 1
        self.epsilon = 0.1

    def hash_state(self, state):
        state = ((np.array(state) + np.array([1.2, 0.07]))/np.array([0.1, 0.01])).astype(int)
        return state[1] * self.individualCounts[0] + state[0]

    # Gets the next action to take given the current state
    # Uses gama to decide when to randomly select an action (Explore) and when to select based on Q value (Exploit)
    def get_action(self, s, best = False):
        s = self.hash_state(s)
        if best or np.random.random() > self.epsilon:
            return np.argmax(self.QFunction[s])
        return np.random.randint(0, self.actionCount)

    # Update Q function
    def update(self, state, action, state_prime, reward):
        h_state = self.hash_state(state)
        h_state_prime = self.hash_state(state_prime)
        self.QFunction[h_state, action] = (1 - self.alpha) * self.QFunction[h_state, action] + self.alpha * (
                    reward + self.gamma * self.QFunction[h_state_prime, self.get_action(state_prime, best=True)])


class Simulation:
    def __init__(self, agent = QLearning):
        self.env = gym.make('MountainCar-v0')
        self.agent = agent(self.env)
        self.env.reset()

        self.R = 0
        self.t = 0

        self.state = self.env.state
        self.reward = 0
        self.done = False

    def reset(self):
        self.env.reset()

        self.R = 0
        self.t = 0

        self.state = self.env.state
        self.reward = 0
        self.done = False

    def simulate(self, train=False):
        while not self.done:
            self.env.render()
            action = self.agent.get_action(self.state, best=not train)
            state_prime, self.reward, self.done, info = self.env.step(action)
            print(action, state_prime)
            self.R = self.R + (self.agent.gamma**self.t)*self.reward

            if train:
                self.agent.update(self.state, action, state_prime, self.reward)
            self.state = state_prime
            self.t += 1

        return self.reward, self.R


if __name__ == "__main__":
    episodes = 5000
    number_of_rollouts = 100
    accuracy_values = []
    R_values = []
    sim = Simulation()
    for i in range(episodes):
        print("--------------------------------------------------Episode", i)
        sim.simulate(train=True)
        sim.reset()
        if i % 1000 == 0:
            accuracy = 0
            R_ave = 0
            for j in range(number_of_rollouts):
                success, R = sim.simulate(train=False)
                sim.reset()
                accuracy += success
                R_ave += R
            accuracy_values.append(accuracy/number_of_rollouts)
            R_values.append(R_ave/number_of_rollouts)
            print("Episode: ", i, " Accuracy: ", accuracy/number_of_rollouts, " R: ", R_ave/number_of_rollouts)

    plt.plot(np.arange(len(accuracy_values)), accuracy_values)
    plt.xlabel('x100 Episodes')
    plt.ylabel('Accuracy')
    plt.show()

    # Expected discounted return
    # stochastic: 0.0688909
    # non-stochastic: 0.59049
    plt.plot(np.arange(len(R_values)), R_values)
    #plt.plot(np.arange(len(R_values)), np.ones(len(R_values))*0.0688909)
    plt.xlabel('x100 Episodes')
    plt.ylabel('Discounted Return')
    plt.show()
