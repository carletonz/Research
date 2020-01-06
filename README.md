
# Q-learning  
### Non-Stochastic Environment
When using the frozen lake environment where is_slippery=False (Non-stochastic) the accuracy of the agent plateaus around 200 to 300 episodes. The sharp increase in accuracy is normal because while testing there is no randomness in this case. This means that the path it takes on the first try while testing is the same path it will take on all other tries.

![](https://github.com/carletonz/Research/raw/master/graphs/Accuracy-Non-Stochastic.png)

For this environment the total reward seems to start to convert to about 5 after 7000 episodes.
 
![](https://github.com/carletonz/Research/raw/master/graphs/R-Non-Stochastic.png)

### Stochastic Environment
When using a stochastic environment the accuracy of the agent seems to start to plateau around 10000 episodes.

![](https://github.com/carletonz/Research/raw/master/graphs/Accracy-Stochastic.png)
![](https://github.com/carletonz/Research/raw/master/graphs/R-Stochastic.png)

---
For the next 4 sections our goal is to show that in certain situations a joint agent can perform better than separate agetns, while in other situations separate agents will perform better than joint agents.

# Joint: Good (Frozen Lake)
### Stochastic Environment
For this example we will be looking at a joint agent (i.e. an agent that chooses 2 actions to do at every time step). We will use the frozen lake environment for this example. One of the actions is to go in the horizontal or verticle direction. The other action is to go in the positive or negative direction. Conceptually this is the same as the stochastic environment in the Q-learning section above. We will use those graphs to look at how well the agent did.

![](https://github.com/carletonz/Research/raw/master/graphs/Accracy-Stochastic.png)

We define accuracy as the number of times the agent got to its goal in the frozen lake environment divided by the total number of attempts. To obtain this graph we averaged the accuracy of the agent over 100 rollouts every 100th episode. As seen in the graph the agent reaches its maximum accuracy (about 75%) around episode 15000.

![](https://github.com/carletonz/Research/raw/master/graphs/R-Stochastic.png)

The blue line in this graph shows the discounted return every 100th episode. The orange line show the expected discounted return, which we calculated using value iteration algorithm. The expected discounted return came out to be 0.0688909.

# Separate: Bad (Frozen Lake)
### Stochastic Environment
For this example we will be looking at separate agents. This means that there will be multiple agents and each agent will choose one action. Similarly to the joint example we will be using the frozen lake environment. Agent 1 will decide if the action should be in the horizontal or vertical direction. Agent2 will decide if the action should be in the positive or negative direction. 

![](https://github.com/carletonz/Research/raw/master/graphs/separate_bad_accuracy_stochastic.png)

We can see that in this example the agents' accuracy only reaches about 30%, which is much lower than what we see in the joint example above.

![](https://github.com/carletonz/Research/raw/master/graphs/separate_bad_R_stochastic.png)

In this example the discounte return only reaches about 0.05. This is also lower than the joint example above.

One reason why the separate agents do worse than the joint agent is because the separate agent takes longer to reach an optimal policy. This would make sence because both agents dont know the action the other agent took. This might make it harder for them to work with each other.

# Joint: Bad (Frozen Lake / Mountain Car)
### Frozen Lake: stochastic environment
### Mountain Car: actions are discrete

(Coming soon)

# Separate: Good (Frozen Lake / Mountain Car)
### Frozen Lake: stochastic environment
### Mountain Car: actions are discrete

(Coming soon)
