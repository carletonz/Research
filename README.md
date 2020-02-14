
# Q-learning  
### Non-Stochastic Environment
When using the frozen lake environment where is_slippery=False (Non-stochastic) the accuracy of the agent plateaus around 200 episodes. The sharp increase in accuracy is normal because while testing there is no randomness in this case. This means that the path it takes on the first try while testing is the same path it will take on all other tries.

![](https://github.com/carletonz/Research/raw/master/graphs/Accuracy-Non-Stochastic.png)

For this environment the discounted return (the blue line) converges to 0.59049 after 200 episodes. We calculated the expected discounted return using value iteration, which is also 0.59049.
 
![](https://github.com/carletonz/Research/raw/master/graphs/R-Non-Stochastic.png)

### Stochastic Environment
When using a stochastic environment the accuracy and discounted return of the agent seems to start to converge around 21000 episodes. The accuracy converges to about 70% and the discounted return converges to about 0.0688909.

![](https://github.com/carletonz/Research/raw/master/Q-Learning/FrozenLake_Accuracy.png)
![](https://github.com/carletonz/Research/raw/master/Q-Learning/FrozenLake_Return.png)

---
For the next 4 sections our goal is to show that in certain situations a joint agent can perform better than separate agetns, while in other situations separate agents will perform better than joint agents.

# Joint: Good (Frozen Lake)
### Stochastic Environment
For this example we will be looking at a joint agent (i.e. an agent that chooses 2 actions to do at every time step). We will use the frozen lake environment for this example. One of the actions is to go in the horizontal or verticle direction. The other action is to go in the positive or negative direction. Conceptually this is the same as the stochastic environment in the Q-learning section above.

![](https://github.com/carletonz/Research/raw/master/Plots_Joint_Good/FrozenLake_Accuracy.png)

We define accuracy as the number of times the agent got to its goal in the frozen lake environment divided by the total number of attempts. To obtain this graph we averaged the accuracy of the agent over 100 rollouts every 100th episode. As seen in the graph the agent reaches its maximum accuracy (about 70%) around episode 15000.

![](https://github.com/carletonz/Research/raw/master/Plots_Joint_Good/FrozenLake_Return.png)

The blue line in this graph shows the discounted return every 100th episode. The orange line show the expected discounted return, which we calculated using value iteration algorithm. The expected discounted return came out to be 0.0688909.

# Separate: Bad (Frozen Lake)
### Stochastic Environment
For this example we will be looking at separate agents. This means that there will be multiple agents and each agent will choose one action. Similarly to the joint example we will be using the frozen lake environment. Agent 1 will decide if the action should be in the horizontal or vertical direction. Agent2 will decide if the action should be in the positive or negative direction. 

![](https://github.com/carletonz/Research/raw/master/graphs/separate_bad_accuracy_stochastic.png)

We can see that in this example the agents' accuracy only reaches about 30%, which is much lower than what we see in the joint example above.

![](https://github.com/carletonz/Research/raw/master/graphs/separate_bad_R_stochastic.png)

In this example the discounte return only reaches about 0.05. This is also lower than the joint example above.

One reason why the separate agents do worse than the joint agent is because the separate agent takes longer to reach an optimal policy. This would make sence because both agents dont know the action the other agent took. This might make it harder for them to work with each other.

# Joint: Bad (Frozen Lake / Taxi)
### Frozen Lake: stochastic environment

![](https://github.com/carletonz/Research/raw/master/Plots_Joint_Bad/FrozenLake_Return.png)
![](https://github.com/carletonz/Research/raw/master/Plots_Joint_Bad/Taxi_Return.png)
![](https://github.com/carletonz/Research/raw/master/Plots_Joint_Bad/FrozenLake_Taxi_Return.png)

# Separate: Good (Frozen Lake / Taxi)
### Frozen Lake: stochastic environment

![](https://github.com/carletonz/Research/blob/master/Plots_Separate_Good/notSummed/FrozenLake_Separate_Good_Return.png)
![](https://github.com/carletonz/Research/blob/master/Plots_Separate_Good/notSummed/Taxi_Separate_Good_Return.png)
![](https://github.com/carletonz/Research/blob/master/Plots_Separate_Good/notSummed/Taxi_FrozenLake_Separate_Good_Return.png)
