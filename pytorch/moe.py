import sac
import gym
import torch
import time
from utils import EnvSet
from sac_core import MLPActorCritic

def create_ant_env():
    return gym.make("Ant-v2")

def create_cheetah_env():
    return gym.make("HalfCheetah-v2")

def create_env():
    return EnvSet([create_ant_env, create_cheetah_env])

sac.sac(create_env, MLPActorCritic, epochs=200, steps_per_epoch=4000, ac_kwargs={"hidden_sizes":(256, 256, 14)} , logger_kwargs={"output_dir": "/home/ubuntu/Documents/proj/research/Research/results/moe-%i"%int(time.time())}, save_gating=True)

