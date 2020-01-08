import gym
import numpy as np
import matplotlib.pyplot as plt


class QLearning:
    def __init__(self, environment1, environment2):

        # position: -1.2 to 0.6
        # speed: -0.07 to 0.07

        # make continuous states as discrete by dividing by 0.1 and 0.01
        # this makes 19 discrete states for position and 15 discrete states for speed
        # with a total of 285 states

        self.stateCounts = ((environment1.high - environment1.low) / np.array([0.1, 0.01])).astype(int)
        self.stateCounts[0] += 2
        self.stateCounts[1] += 1
        self.stateCounts = np.append(self.stateCounts, environment2.nS)
        self.actionCount = np.array([environment1.action_space.n, environment2.nA])

        self.QFunction = np.zeros((self.stateCounts[0], self.stateCounts[1], self.stateCounts[2], self.actionCount[0], self.actionCount[1]))
        self.alpha = 0.05 # 0.01
        self.gamma = 1
        self.epsilon = 0.1

    def hash_state(self, state):
        return ((np.array(state) + np.array([1.2, 0.07]))/np.array([0.1, 0.01])).astype(int)

    # Gets the next action to take given the current state
    # Uses gama to decide when to randomly select an action (Explore) and when to select based on Q value (Exploit)
    def get_action(self, s1, s2, best = False):
        s = self.hash_state(s1)
        if best or np.random.random() > self.epsilon:
            return np.unravel_index(np.argmax(self.QFunction[s[0], s[1], s2]), self.QFunction[s[0], s[1], s2].shape)
        return np.array([np.random.randint(0, self.actionCount[0]), np.random.randint(0, self.actionCount[1])])

    # Update Q function
    def update(self, state1, action1, state_prime1, reward1, state2, action2, state_prime2, reward2):
        h_state = self.hash_state(state1)
        h_state_prime = self.hash_state(state_prime1)
        best_action = self.get_action(state_prime1, state_prime2, best=True)

        self.QFunction[h_state[0], h_state[1], state2, action1, action2] = (1 - self.alpha) * self.QFunction[h_state[0], h_state[1], state2, action1, action2] + self.alpha * (
                    0.005*reward1 + reward2 + self.gamma * self.QFunction[h_state_prime[0], h_state_prime[1], state_prime2, best_action[0], best_action[1]])


class Simulation:
    def __init__(self, agent = QLearning):
        self.env1 = gym.make('MountainCar-v0')
        self.env2 = gym.make('FrozenLake-v0', is_slippery=True)
        self.agent = agent(self.env1, self.env2)

        self.R1 = 0
        self.t1 = 0

        self.state1 = self.env1.reset()
        self.reward1 = 0
        self.done1 = False

        self.R2 = 0
        self.t2 = 0

        self.env2.reset()
        self.state2 = self.env2.s
        self.reward2 = 0
        self.done2 = False

    def reset(self):
        self.R1 = 0
        self.t1 = 0

        self.state1 = self.env1.reset()
        self.reward1 = 0
        self.done1 = False

        self.R2 = 0
        self.t2 = 0

        self.env2.reset()
        self.state2 = self.env2.s
        self.reward2 = 0
        self.done2 = False

    def simulate(self, train=False):
        while not self.done1 or not self.done2:
            if not train:
                self.env1.render()

            action = self.agent.get_action(self.state1, self.state2, best=not train)

            if not self.done1:
                self.state_prime1, self.reward1, self.done1, _ = self.env1.step(action[0])
                self.R1 = self.R1 + (self.agent.gamma ** self.t1) * self.reward1
            if not self.done2:
                self.state_prime2, self.reward2, self.done2, _ = self.env2.step(action[1])
                self.R2 = self.R2 + (self.agent.gamma ** self.t2) * self.reward2

            if train:
                self.agent.update(self.state1, action[0], self.state_prime1, self.reward1, self.state2, action[1], self.state_prime2, self.reward2)
            self.state1 = self.state_prime1
            self.state2 = self.state_prime2
            self.t1 += 1
            self.t2 += 1
        return self.reward1, self.R1, self.reward2, self.R2


if __name__ == "__main__":
    episodes = 50000
    number_of_rollouts = 20
    accuracy_values1 = []
    R_values1 = []

    accuracy_values2 = []
    R_values2 = []
    sim = Simulation()
    for i in range(episodes):
        print("--------------------------------------------------Episode", i)
        sim.simulate(train=True)
        sim.reset()
        if (i+1) % 1000 == 0:
            accuracy1 = 0
            R_ave1 = 0
            accuracy2 = 0
            R_ave2 = 0
            for j in range(number_of_rollouts):
                success1, R1, success2, R2 = sim.simulate(train=False)
                sim.reset()
                accuracy1 += success1
                R_ave1 += R1

                accuracy2 += success2
                R_ave2 += R2
            accuracy_values1.append(accuracy1/number_of_rollouts)
            R_values1.append(R_ave1/number_of_rollouts)

            accuracy_values2.append(accuracy2 / number_of_rollouts)
            R_values2.append(R_ave2 / number_of_rollouts)
            print("Episode: ", i, " Accuracy: ", accuracy1/number_of_rollouts, " R: ", R_ave1/number_of_rollouts)

    plt.plot(np.arange(len(accuracy_values1)), accuracy_values1)
    plt.xlabel('x1000 Episodes (Mountain Car)')
    plt.ylabel('Accuracy')
    plt.show()

    plt.plot(np.arange(len(R_values1)), R_values1)
    #plt.plot(np.arange(len(R_values)), np.ones(len(R_values))*0.0688909)
    plt.xlabel('x1000 Episodes (Mountain Car)')
    plt.ylabel('Discounted Return')
    plt.show()

    plt.plot(np.arange(len(accuracy_values2)), accuracy_values2)
    plt.xlabel('x1000 Episodes (Frozen Lake)')
    plt.ylabel('Accuracy')
    plt.show()

    plt.plot(np.arange(len(R_values2)), R_values2)
    # plt.plot(np.arange(len(R_values)), np.ones(len(R_values))*0.0688909)
    plt.xlabel('x1000 Episodes (Frozen Lake)')
    plt.ylabel('Discounted Return')
    plt.show()
