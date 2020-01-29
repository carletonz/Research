import gym
import numpy as np
import matplotlib.pyplot as plt


class QLearning:
    def __init__(self, environment):
        self.actionCount = environment.nA
        self.QFunction = np.zeros((environment.nS, self.actionCount))
        self.alpha = 0.007
        self.gamma = 0.9
        self.epsilon = 0.2

    # Gets the next action to take given the current state
    # Uses gama to decide when to randomly select an action (Explore) and when to select based on Q value (Exploit)
    def get_action(self, s, best = False):
        if best or np.random.random() > self.epsilon:
            return np.argmax(self.QFunction[s])
        return np.random.randint(0, self.actionCount)

    # Update Q function
    def update(self, state, action, state_prime, reward):
        self.QFunction[state, action] = (1 - self.alpha) * self.QFunction[state, action] + self.alpha * (
                    reward + self.gamma * self.QFunction[state_prime, self.get_action(state_prime, best=True)])


class Simulation:
    def __init__(self, agent = QLearning):
        # environment
        # is_slippery=True --> 33% any action except the opposite of the chosen action is done
        # is_slippery=False --> 100% chance chosen action is taken
        self.env = gym.make('FrozenLake-v0', is_slippery=True)
        self.agent = agent(self.env)

        self.R = 0
        self.t = 0

        self.state = self.env.s
        self.reward = 0
        self.done = False

        self.env.reset()

    def reset(self):
        self.env.reset()
        self.R = 0
        self.t = 0

        self.state = self.env.s
        self.reward = 0
        self.done = False

    def simulate(self, train=False):
        while not self.done:
            action = self.agent.get_action(self.state, best=not train)
            state_prime, self.reward, self.done, info = self.env.step(action)

            self.R = self.R + (self.agent.gamma**self.t)*self.reward

            if train:
                self.agent.update(self.state, action, state_prime, self.reward)
            self.state = state_prime
            self.t += 1

        return self.reward, self.R


if __name__ == "__main__":
    episodes = 50000
    number_of_rollouts = 1000
    accuracy_values = []
    R_values = []
    sim = Simulation()
    for i in range(episodes):
        sim.simulate(train=True)
        sim.reset()
        if i % 100 == 0:
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
    plt.plot(np.arange(len(R_values)), np.ones(len(R_values))*0.0688909)
    plt.xlabel('x100 Episodes')
    plt.ylabel('Discounted Return')
    plt.show()
