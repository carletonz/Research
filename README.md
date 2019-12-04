
# Q-learning  
### Non-Stochastic Environment
When using the frozen lake environment where is_slippery=False (Non-stochastic) the accuracy of the agent plateaus around 200 to 300 episodes. The sharp increase in accuracy is normal because while testing there is no randomness in this case. This means that the path it takes on the first try while testing is the same path it will take on all other tries.

![](https://github.com/carletonz/Research/raw/master/Accuracy-Non-Stochastic.png)
For this environment the total reward seems to start to convert to about 5 after 7000 episodes.
 
![](https://github.com/carletonz/Research/raw/master/Total_Reward-Non-Stochastic.png)

### Stochastic Environment
When using a stochastic environment the accuracy of the agent seems to start to plateau around 10000 episodes.
![](https://github.com/carletonz/Research/raw/master/Accracy-Stochastic.png)
![](https://github.com/carletonz/Research/raw/master/Total_Reward-Stochastic.png)

# Separate: Bad (frozen lake)
### Non-Stochastic Environment
Graph of accuracy. For this model there were 2 agents. Agent 1 decides if the action should be in the horizontal or vertical direction. Agent2 decides if the action should be in the positive or negative direction given agent 1's action.
![](https://github.com/carletonz/Research/raw/master/separate_bad_accuracy_non-stochastic.png)
![](https://github.com/carletonz/Research/raw/master/separate_bad_total_reward1_non-stochastic.png)
![](https://github.com/carletonz/Research/raw/master/separate_bad_total_reward2_non-stochastic.png)

Also tried a model where agent 2 does not know the action of agent 1. In this case it was significantly harder to get total reward to converge. (Graphs coming soon)

# Joint: Good
### Non-Stochastic Environment
conceptually this should be the same as the non-stochastic environment in the Q-learning section above.

![](https://github.com/carletonz/Research/raw/master/joint_good_accuracy_non-stochastic.png)
![](https://github.com/carletonz/Research/raw/master/joint_good_total_reward_non-stochastic.png)
