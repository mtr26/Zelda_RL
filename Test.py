from Rom.Zelda_env import *
from stable_baselines3.common.vec_env import VecFrameStack, VecMonitor, SubprocVecEnv, DummyVecEnv, VecTransposeImage
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from Rom.SaveOnBestCallback import SaveOnBestTrainingRewardCallback
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import plot_results
import os
import matplotlib.pyplot as plt
from Train import ZeldaFeatureExtractor


def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = ZeldaEnv(rank, save=False,show=True, speed=8)
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init


import argparse

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--timesteps', type=int, default=int(1e6))
    argparser.add_argument('--model_path')

    args = argparser.parse_args()
    timesteps = args.timesteps
    model_path = args.model_path


    vec_env = DummyVecEnv([make_env(0)])
    vec_env = VecFrameStack(vec_env, n_stack=4)
    vec_env = VecTransposeImage(vec_env)

    # Workaround for cloudpickle failing to load policy_kwargs from a torch.compiled model
    custom_objects = {
        "policy_kwargs": dict(features_extractor_class=ZeldaFeatureExtractor)
    }
    model = PPO.load(model_path, env=vec_env, custom_objects=custom_objects)
    obs = vec_env.reset()

    for _ in range(timesteps):
        action, _state = model.predict(obs)
        obs, reward, done, info = vec_env.step(action)


