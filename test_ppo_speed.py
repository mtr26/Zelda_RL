import time
import torch
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from Rom.Zelda_env import ZeldaEnv

def make_env(rank):
    def _init():
        env = ZeldaEnv(pos=0, save=False, show=False)
        return env
    return _init

if __name__ == '__main__':
    # Test with default torch threads
    vec_env = SubprocVecEnv([make_env(i) for i in range(4)])
    model = PPO('CnnPolicy', vec_env, n_steps=256, batch_size=128, n_epochs=1)
    
    start = time.time()
    model.learn(total_timesteps=1024)
    end = time.time()
    print(f"Default threads: {1024 / (end - start):.2f} steps/sec")
    
    # Test with 1 torch thread
    torch.set_num_threads(1)
    os.environ['OMP_NUM_THREADS'] = '1'
    model2 = PPO('CnnPolicy', vec_env, n_steps=256, batch_size=128, n_epochs=1)
    
    start = time.time()
    model2.learn(total_timesteps=1024)
    end = time.time()
    print(f"1 Torch thread: {1024 / (end - start):.2f} steps/sec")
