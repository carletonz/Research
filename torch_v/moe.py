import spinup
import gym
import torch
from utils import EnvSet
from sac_core import MLPActorCritic

def create_ant_env():
    return gym.make("Ant-v2")

def create_cheetah_env():
    return gym.make("HalfCheetah-v2")

def create_env():
    return EnvSet([create_ant_env, create_cheetah_env])

spinup.sac_pytorch(create_env, MLPActorCritic, epochs=5)

