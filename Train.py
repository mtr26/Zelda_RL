from Rom.Zelda_env import *
from stable_baselines3.common.vec_env import VecFrameStack, VecMonitor, SubprocVecEnv, DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from Rom.SaveOnBestCallback import SaveOnBestTrainingRewardCallback
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import plot_results
import os
import matplotlib.pyplot as plt



def make_env(rank, seed=0, max_time = 2048 * 8):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = ZeldaEnv(rank, save=False, max_step=max_time)
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init


import argparse
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--timesteps', type=int, default=int(1e6))
    argparser.add_argument('--num_cpu', type=int, default=10)
    argparser.add_argument('--log_dir', type=str, default="tmp/")
    argparser.add_argument('--pre_trained', type=bool, default=False)

    args = argparser.parse_args()
    timesteps = args.timesteps
    num_cpu = args.num_cpu
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    pre_trained = args.pre_trained

    end_model = True
    ep_length = 1024 * 8

    vec_env = SubprocVecEnv([make_env(i, ep_length) for i in range(num_cpu)])


    vec_env = VecFrameStack(vec_env, n_stack=4)

    vec_env = VecMonitor(vec_env, log_dir)
    callback = SaveOnBestTrainingRewardCallback(check_freq=256, log_dir=log_dir)

    
    if pre_trained is not False:
        model = PPO.load('end_model' if end_model else 'best_model', env=vec_env)
        model.set_parameters('end_model' if end_model else 'best_model')
        model.rollout_buffer.buffer_size = ep_length
        model.rollout_buffer.reset()
    else:
        model = PPO('CnnPolicy', env=vec_env,  n_steps=ep_length, batch_size=512, n_epochs=1, gamma=0.999)
        
    model.learn(total_timesteps=timesteps, progress_bar=True, callback=callback)
    model.save('end_model')

    plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "ZeldaTest")
    plt.show()
    
