from Rom.Zelda_env import *
from stable_baselines3.common.vec_env import VecFrameStack, VecMonitor, SubprocVecEnv, DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from Rom.SaveOnBestCallback import SaveOnBestTrainingRewardCallback
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import plot_results
import os
import matplotlib.pyplot as plt
import keyboard


def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = ZeldaEnv(rank, save=False,show=True, speed=2)
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init

def inp():
    curent = 26
    if keyboard.is_pressed('z'):
        curent = 3

    elif keyboard.is_pressed('s'):
        curent = 0

    elif keyboard.is_pressed('q'):
        curent = 1

    elif keyboard.is_pressed('d'):
        curent = 2


    elif keyboard.is_pressed('a'):
        curent = 4


    elif keyboard.is_pressed('e'):
        curent = 5

    elif keyboard.is_pressed('v'):
        curent = 'start'


    return curent


if __name__ == '__main__':
    vec_env = SubprocVecEnv([make_env(i) for i in range(1)])
    vec_env = VecFrameStack(vec_env, n_stack=4)

    model = PPO.load('best_model', env=vec_env)
    obs = vec_env.reset()

    for _ in range(int(1e4)):
        action = inp()
        obs, reward, done, info = vec_env.step([action])


