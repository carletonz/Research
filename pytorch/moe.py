import sac
import gym
import torch
import time
from utils import EnvSet, EnvDecomposed, run_model
from sac_core import MLPActorCritic
import numpy as np

def create_ant_env():
    return gym.make("Ant-v2")

def create_cheetah_env():
    return gym.make("HalfCheetah-v2")

def create_env(video_path = None):
    return EnvDecomposed(create_cheetah_env, [3,3], np.arange(6))

def run():
    sac.sac(create_env, 
            MLPActorCritic, 
            epochs=200, 
            steps_per_epoch=4000, 
            ac_kwargs={"hidden_sizes":(256, 256, 6)}, 
            logger_kwargs={"output_dir": "/home/ubuntu/Documents/proj/research/Research/results/moe-decomposed-%i"%int(time.time())},
            save_gating=True)

def test():
    res_id = 1589012680
    result_path = "/home/ubuntu/Documents/proj/research/Research/results/moe-"
    video_path = [result_path+str(res_id)+"/videoEnv0/",None]
    model_path = result_path+str(res_id)+"/pyt_save/model.pt"
    run_model(create_env, model_path, video_path)

run()
