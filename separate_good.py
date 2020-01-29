import gym
import numpy as np
import matplotlib.pyplot as plt


class QLearning:
    def __init__(self, alpha, gamma, epsilon):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
    
    def setEnv(self, environment):
        self.actionCount = environment.nA
        self.QFunction = np.zeros((environment.nS, self.actionCount))
    
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
        
class EpisodeSimulation:
    def __init__(self, env, agent):
        self.env1 = env
        self.agent1 = agent
        self.agent1.setEnv(self.env1)
        
        self.reset()
        self.starting_state = self.state1

    def reset(self):
        self.R1 = 0
        self.t1 = 0
        
        self.state1 = self.env1.reset()
        self.state_prime1 = None
        self.reward1 = 0
        self.done1 = False
        
        self.action1 = None
    
    def take_action(self, train=True, done = False):
        if done:
            return self.reward1, self.done1
        #if not train:
        #    self.env1.render()

        self.action1 = self.agent1.get_action(self.state1, best=not train)
        self.state_prime1, self.reward1, self.done1, info1 = self.env1.step(self.action1)
        self.R1 = self.R1 + (self.agent1.gamma ** self.t1) * self.reward1
        
        return self.reward1, self.done1
    
    def learn(self, reward, train = True, done = False):
        if done:
            return
        
        if train:
            self.agent1.update(self.state1, self.action1, self.state_prime1, self.reward1)
        self.state1 = self.state_prime1
        self.t1 += 1
    
    def get_stats(self):
        return self.reward1, self.R1

def run_episode(sim, train=True):
    done = [False for i in range(len(sim))]
    reward = [0 for i in range(len(sim))]
    while not all(done):
        done_temp = [done[i] for i in range(len(sim))]
        for i in range(len(sim)):
            r, d = sim[i].take_action(train, done[i])
            done_temp[i] = d
            reward[i] = r
        for i in range(len(sim)):
            sim[i].learn(sum(reward), train, done[i])
        done = done_temp
    temp = np.array([sim[i].get_stats() for i in range(len(sim))]).flatten()
    for s in sim:
        s.reset()
    return temp

if __name__ == "__main__":
    episodes = 100000
    number_of_rollouts = 100
    test_after = 1000 #episodes
    acc_r_ave = np.zeros((int(episodes/test_after), 4))
    
    sim = [EpisodeSimulation(gym.make("FrozenLake-v0"), QLearning(0.01, 0.9, 0.07)),
           EpisodeSimulation(gym.make("Taxi-v3"), QLearning(0.01, 0.9, 0.02))]
    
    for i in range(episodes):
        print("--------------------------------------------------Episode", i)
        run_episode(sim)
        if (i+1) % test_after == 0:
            acc_r = np.zeros((4,))
            for j in range(number_of_rollouts):
                acc_r += run_episode(sim, train=False)
            acc_r /= number_of_rollouts
            acc_r_ave[int((i+1) / test_after)-1] = acc_r
            print("Episode: ", i, " Accuracy: ", acc_r[0], " R: ", acc_r[1])
    
    plt.plot(np.arange(len(acc_r_ave[:,0].flatten())), acc_r_ave[:,0].flatten())
    plt.xlabel('x1000 Episodes')
    plt.ylabel('Accuracy')
    plt.savefig("Research/plots2/FL_SG_Accuracy__rewards_summed" + str(np.random.randint(10000)) + ".png")
    plt.clf()
    
    
    plt.plot(np.arange(len(acc_r_ave[:,1].flatten())), acc_r_ave[:,1].flatten())
    plt.plot(np.arange(len(acc_r_ave[:,1].flatten())), np.ones(len(acc_r_ave[:,1].flatten()))*0.0688909)
    plt.xlabel('x1000 Episodes')
    plt.ylabel('Discounted Return')
    plt.savefig("Research/plots2/FL_SG_R__rewards_summed" + str(np.random.randint(10000)) + ".png")
    plt.clf()
    
    plt.plot(np.arange(len(acc_r_ave[:,2].flatten())), acc_r_ave[:,2].flatten())
    plt.xlabel('x1000 Episodes')
    plt.ylabel('Accuracy')
    plt.clf()

    taxi_vi = np.load("Research/Taxi_value_itteration.npy")
    print(sim[1].starting_state)
    print(taxi_vi[sim[1].starting_state])
    
    plt.plot(np.arange(len(acc_r_ave[:,3].flatten())), acc_r_ave[:,3].flatten())
    plt.plot(np.arange(len(acc_r_ave[:,3].flatten())), np.ones(len(acc_r_ave[:,3].flatten()))*taxi_vi[sim[1].starting_state])
    plt.xlabel('x1000 Episodes')
    plt.ylabel('Discounted Return')
    plt.savefig("Research/plots2/Taxi_SG_R__rewards_summed" + str(np.random.randint(10000)) + ".png")
    plt.clf()
    
    np.save("Research/data5/separate_good_rewards_summed.npy", acc_r_ave)