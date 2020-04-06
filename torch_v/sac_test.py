import spinup
import gym

def create_env():
    return gym.make('Humanoid-v2')

spinup.sac_pytorch(create_env, epochs=5)
