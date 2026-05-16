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


def make_env(rank, seed=0, record=False):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        # If recording, hide the window and enable saving
        env = ZeldaEnv(rank, save=record, show=not record, speed=8 if not record else 0)
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init


import argparse

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--timesteps', type=int, default=int(1e6))
    argparser.add_argument('--model_path', required=True, help="Path to the model zip file")
    argparser.add_argument('--mp4', action='store_true', help="Generate an MP4 video instead of opening a window")

    args = argparser.parse_args()
    timesteps = args.timesteps
    model_path = args.model_path
    record = args.mp4


    vec_env = DummyVecEnv([make_env(0, record=record)])
    vec_env = VecFrameStack(vec_env, n_stack=4)
    vec_env = VecTransposeImage(vec_env)

    # Workaround for cloudpickle failing to load policy_kwargs from a torch.compiled model
    custom_objects = {
        "policy_kwargs": dict(features_extractor_class=ZeldaFeatureExtractor)
    }
    model = PPO.load(model_path, env=vec_env, custom_objects=custom_objects)
    obs = vec_env.reset()

    print(f"[INFO] Starting Test run. Recording: {record}")
    
    # If recording, only run until the first episode ends to save a clean video
    steps = 0
    while steps < timesteps:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        steps += 1
        
        if record and done[0]:
            print(f"[INFO] Episode finished after {steps} steps. Saving video...")
            break

    vec_env.close()
    if record:
        print("[SUCCESS] MP4 Video has been saved to the vid/ directory!")


