import gym
import numpy as np
import matplotlib.pyplot as plt


class QLearningFL:
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

class QLearningMC:
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
        self.alpha = 0.05 # 0.01
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
    def __init__(self, agent1 = QLearningMC, agent2 = QLearningFL):
        self.env1 = gym.make('MountainCar-v0')
        self.env2 = gym.make('FrozenLake-v0', is_slippery=True)
        self.agent1 = agent1(self.env1)
        self.agent2 = agent2(self.env2)

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

            if not self.done1:
                action1 = self.agent1.get_action(self.state1, best=not train)
                self.state_prime1, self.reward1, self.done1, info1 = self.env1.step(action1)
                self.R1 = self.R1 + (self.agent1.gamma ** self.t1) * self.reward1

            if not self.done2:
                action2 = self.agent2.get_action(self.state2, best=not train)
                self.state_prime2, self.reward2, self.done2, info2 = self.env2.step(action2)
                self.R2 = self.R2 + (self.agent2.gamma ** self.t2) * self.reward2

            if train:
                if not self.done1:
                    self.agent1.update(self.state1, action1, self.state_prime1, self.reward1+self.reward2)
                if not self.done2:
                    self.agent2.update(self.state2, action2, self.state_prime2, self.reward1+self.reward2)
            self.state1 = self.state_prime1
            self.t1 += 1

            self.state2 = self.state_prime2
            self.t2 += 1

        return self.reward1, self.reward2, self.R1, self.R2, self.R1 + self.R2


if __name__ == "__main__":
    episodes = 50000
    number_of_rollouts = 20
    accuracy_values = []
    R_values1 = []
    R_values2 = []
    sim = Simulation()
    for i in range(episodes):
        print("--------------------------------------------------Episode", i)
        sim.simulate(train=True)
        sim.reset()
        if (i+1) % 1000 == 0:
            accuracy = 0
            R_ave1 = 0
            R_ave2 = 0
            for j in range(number_of_rollouts):
                success1,  success2, R1, R2, Rtotal = sim.simulate(train=False)
                sim.reset()
                accuracy += success2
                R_ave1 += R1
                R_ave2 += R2
            accuracy_values.append(accuracy/number_of_rollouts)
            R_values1.append(R_ave1 / number_of_rollouts)
            R_values2.append(R_ave2 / number_of_rollouts)
            print("Episode: ", i, " Accuracy: ", accuracy/number_of_rollouts, " R: ", R_ave1/number_of_rollouts)

    plt.plot(np.arange(len(accuracy_values)), accuracy_values)
    plt.xlabel('x100 Episodes')
    plt.ylabel('Accuracy')
    plt.show()

    plt.plot(np.arange(len(R_values1)), R_values1)
    plt.xlabel('x1000 Episodes')
    plt.ylabel('Discounted Return')
    plt.savefig("Research/plots/MC_SG_R_" + str(np.random.randint(10000)) + ".png")

    plt.plot(np.arange(len(R_values2)), R_values2)
    plt.xlabel('x1000 Episodes')
    plt.ylabel('Discounted Return')
    plt.savefig("Research/plots/FL_SG_R_" + str(np.random.randint(10000)) + ".png")
