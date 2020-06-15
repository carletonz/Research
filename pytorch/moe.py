import sac
import gym
import torch
import time
from utils import EnvSet, EnvDecomposed, run_model, NullEnv
from sac_core import MLPActorCritic
import numpy as np

def create_ant_env():
    return gym.make("Ant-v2")

def create_cheetah_env():
    return gym.make("HalfCheetah-v2")

def create_null_env(env_fn):
    def create_env():
        return NullEnv(env_fn)
    return create_env

def create_env(video_path = None):
    return EnvSet([create_ant_env, create_null_env(create_cheetah_env)],video_path)

def run():
    sac.sac(create_env, 
            MLPActorCritic, 
            epochs=200, 
            steps_per_epoch=4000, 
            ac_kwargs={"hidden_sizes":(256, 256, 14)}, 
            logger_kwargs={"output_dir": "/home/ubuntu/Documents/proj/research/Research/results/moe-%i"%int(time.time())},
            save_gating=True)

def test():
    res_id = 1591049489
    result_path = "/home/ubuntu/Documents/proj/research/Research/results/moe-ant-only-"
    video_path = [result_path+str(res_id)+"/videoEnv0/", None]
    model_path = result_path+str(res_id)+"/pyt_save/model.pt"
    run_model(create_env, model_path, video_path)

run()
