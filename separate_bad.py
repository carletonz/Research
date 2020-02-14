import gym
import numpy as np
import matplotlib.pyplot as plt


class QLearning:
    def __init__(self, environment):
        self.actionCount = int(environment.nA/2)
        self.QFunction = np.zeros((environment.nS, self.actionCount))
        self.alpha = 0.01
        self.gamma = 0.9
        self.epsilon = 0.2
    
    def reset(self):
        self.QFunction = np.zeros(self.QFunction.shape)
    
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
        # is_slippery=True --> 33% chance chosen action is taken, Default
        # is_slippery=False --> 100% chance chosen action is taken
        self.env = gym.make('FrozenLake-v0', is_slippery=True)
        self.agent1 = agent(self.env)  # horizontal or vertical action
        self.agent2 = agent(self.env)  # positive or negative direction

        self.R = 0
        self.t = 0

        self.state = self.env.s
        self.reward = 0
        self.done = False

        self.env.reset()
    
    def resetEverything(self):
        self.agent1.reset()
        self.agent2.reset()
        self.reset()
    
    def reset(self):
        self.env.reset()
        self.R = 0
        self.t = 0

        self.state = self.env.s
        self.reward = 0
        self.done = False

    def simulate(self, train=False):
        while not self.done:
            action1 = self.agent1.get_action(self.state, best=not train)  # horizontal (0) or vertical (1) action
            action2 = self.agent2.get_action(self.state, best=not train)  # negative (0) or positive (1) direction
            state_prime, self.reward, self.done, info = self.env.step(action1 + 2*action2)

            self.R = self.R + (self.agent1.gamma**self.t)*self.reward

            if train:
                self.agent1.update(self.state, action1, state_prime, self.reward)
                self.agent2.update(self.state, action2, state_prime, self.reward)
            self.state = state_prime
            self.t += 1

        return self.reward, self.R


if __name__ == "__main__":
    episodes = 50000
    number_of_rollouts = 100
    test_after = 100 #episodes
    retrain = 20 #times
    
    acc_r_ave = np.zeros((int(episodes/test_after), 2))
    sim = Simulation()
    
    for p in range(retrain):
        for i in range(episodes):
            sim.simulate(train=True)
            sim.reset()
            if (i+1) % test_after == 0:
                acc_r = np.zeros((2))
                for j in range(number_of_rollouts):
                    acc_r += np.array(sim.simulate(train=False))
                    sim.reset()
                acc_r /= number_of_rollouts
                acc_r_ave[int((i+1) / test_after)-1] += acc_r
                print("Itteration: ", p, "Episode: ", i)
                print("FrozenLake | Accuracy: ", acc_r[0], " R: ", acc_r[1])
                print("--------------------------------------------------------")
                print()
        sim.resetEverything()
    
    acc_r_ave = acc_r_ave/retrain

    plt.plot(np.arange(len(acc_r_ave[:, 0].flatten())), acc_r_ave[:, 0].flatten())
    plt.xlabel('x100 Episodes')
    plt.ylabel('Accuracy')
    plt.savefig("Plots_Separate_Bad/FrozenLake_Accruacy.png")
    plt.clf()

    # Expected discounted return
    # stochastic: 0.0688909
    # non-stochastic: 0.59049
    plt.plot(np.arange(len(acc_r_ave[:, 1].flatten())), acc_r_ave[:, 1].flatten())
    plt.plot(np.arange(len(acc_r_ave[:, 1].flatten())), np.ones(len(acc_r_ave[:, 1].flatten()))*0.0688909)
    plt.xlabel('x100 Episodes')
    plt.ylabel('Discounted Return')
    plt.savefig("Plots_Joint_Good/FrozenLake_Return.png")
    plt.clf()
    
    np.save("Data/separate_bad.npy", acc_r_ave)
