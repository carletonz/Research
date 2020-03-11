import gym

# set of observation or action spaces
class SpaceSet:
    def __init__(self, envs):
        
# set of environments
class EnvSet:
    def __init__(self, env_fns):
        self.envs = [env_fns[i]() for i in range(len(env_fns))]
        