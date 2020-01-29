import gym
import numpy as np
import matplotlib.pyplot as plt


class QLearning:
    def __init__(self, environment1, environment2):
        self.actionCount = np.array([environment1.nA, environment2.nA])

        self.QFunction = np.zeros((environment1.nS, environment2.nS, environment1.nA, environment2.nA))
        self.alpha = 0.01
        self.gamma = 0.9
        self.epsilon = 0.01

    # Gets the next action to take given the current state
    # Uses gama to decide when to randomly select an action (Explore) and when to select based on Q value (Exploit)
    def get_action(self, s1, s2, best = False):
        if best or np.random.random() > self.epsilon:
            return np.unravel_index(np.argmax(self.QFunction[s1, s2]), self.QFunction[s1, s2].shape)
        return np.array([np.random.randint(0, self.actionCount[0]), np.random.randint(0, self.actionCount[1])])

    # Update Q function
    def update(self, state1, action1, state_prime1, reward1, state2, action2, state_prime2, reward2):
        best_action = self.get_action(state_prime1, state_prime2, best=True)

        self.QFunction[state1, state2, action1, action2] = (1 - self.alpha) * self.QFunction[state1, state2, action1, action2] + self.alpha * (
                    reward1 + reward2 + self.gamma * self.QFunction[state_prime1, state_prime2, best_action[0], best_action[1]])


class Simulation:
    def __init__(self, agent = QLearning):
        self.env1 = gym.make('Taxi-v3')
        self.env2 = gym.make('FrozenLake-v0', is_slippery=True)
        self.agent = agent(self.env1, self.env2)

        self.reset()

    def reset(self):
        self.R1 = 0
        self.t1 = 0

        self.state1 = self.env1.reset()
        self.reward1 = 0
        self.done1 = False

        self.R2 = 0
        self.t2 = 0

        self.state2 = self.env2.reset()
        self.reward2 = 0
        self.done2 = False

    def simulate(self, train=False):
        while not self.done1 or not self.done2:
            #if not train:
            #    self.env1.render()

            action = self.agent.get_action(self.state1, self.state2, best=not train)

            if not self.done1:
                self.state_prime1, self.reward1, self.done1, _ = self.env1.step(action[0])
                self.R1 = self.R1 + (self.agent.gamma ** self.t1) * self.reward1 #change gamma to 0.9 to correct scaling?
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
    episodes = 100000
    number_of_rollouts = 100
    test_after = 1000 #episodes
    acc_r_ave = np.zeros((int(episodes/test_after), 4))
    sim = Simulation()
    for i in range(episodes):
        print("--------------------------------------------------Episode", i)
        sim.simulate(train=True)
        sim.reset()
        if (i+1) % test_after == 0:
            acc_r = np.zeros((4))
            for j in range(number_of_rollouts):
                acc_r += np.array(sim.simulate(train=False))
                sim.reset()
            acc_r /= number_of_rollouts
            acc_r_ave[int((i+1) / test_after)-1] = acc_r
            print("Episode: ", i, " Accuracy: ", acc_r[0], " R: ", acc_r[1])

    plt.plot(np.arange(len(acc_r_ave[:, 0].flatten())), acc_r_ave[:, 0].flatten())
    plt.xlabel('x1000 Episodes (Taxi)')
    plt.ylabel('Accuracy')
    plt.savefig("Research/plots2/Taxi_JB_A_.png")
    plt.clf()
    
    plt.plot(np.arange(len(acc_r_ave[:, 1].flatten())), acc_r_ave[:, 1].flatten())
    #plt.plot(np.arange(len(R_values)), np.ones(len(R_values))*0.0688909)
    plt.xlabel('x1000 Episodes (Taxi)')
    plt.ylabel('Discounted Return')
    plt.savefig("Research/plots2/Taxi_JB_R_.png")
    plt.clf()
    
    plt.plot(np.arange(len(acc_r_ave[:, 2].flatten())), acc_r_ave[:, 2].flatten())
    plt.xlabel('x1000 Episodes (Frozen Lake)')
    plt.ylabel('Accuracy')
    plt.savefig("Research/plots2/FL_JB_A_.png")
    plt.clf()
    
    plt.plot(np.arange(len(acc_r_ave[:, 3].flatten())), acc_r_ave[:, 3].flatten())
    # plt.plot(np.arange(len(R_value)), np.ones(len(R_values))*0.0688909)
    plt.xlabel('x1000 Episodes (Frozen Lake)')
    plt.ylabel('Discounted Return')
    plt.savefig("Research/plots2/FL_JB_R_.png")
    plt.clf()
    
    plt.plot(np.arange(len(acc_r_ave[:, 1].flatten())), (acc_r_ave[:, 1].flatten()*0.000175)+0.07, label="Taxi")
    plt.plot(np.arange(len(acc_r_ave[:, 3].flatten())), acc_r_ave[:, 3].flatten(), label="Frozen Lake")
    plt.legend()
    plt.xlabel('x1000 Episodes (Frozen Lake / Taxi)')
    plt.ylabel('Discounted Return')
    plt.savefig("Research/plots2/FLMC_JB_R_.png")
    plt.clf()
    
    np.save("Research/data5/joint_bad.npy", acc_r_ave)