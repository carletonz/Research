import spinup
import gym
from utils import EnvSet
import time
import sac
def create_ant_env():
    return gym.make("Ant-v2")

def create_cheetah_env():
    return gym.make("HalfCheetah-v2")

def create_env():
    return EnvSet([create_ant_env, create_cheetah_env])

sac.sac(create_env, epochs=200, steps_per_epoch=4000, logger_kwargs={"output_dir": "/home/ubuntu/Documents/proj/research/Research/results/baseline-%i"%int(time.time())}, baseline=True)

