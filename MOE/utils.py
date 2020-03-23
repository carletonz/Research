from gym import Env

# set of observation or action spaces
class SpaceSet:
    def __init__(self, envs):
        pass
# set of environments
class EnvSet(Env):
    def __init__(self, env_fns):
        self.envs = [env_fns[i]() for i in range(len(env_fns))]
        
        self.action_space = None
        self.observation_space = None
        self.reward_range = None
        
    def step(self):
        pass
    
    def reset(self):
        return [env.reset() for env in self.envs]

    def render(self):
        pass
    
    def close(self):
        pass
    
    def seed(self):
        pass

def temp():
    return 1
EnvSet([temp])