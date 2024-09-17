from Zelda_env import *
from stable_baselines3.common.vec_env import VecFrameStack, VecMonitor, SubprocVecEnv, DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = ZeldaEnv(rank, show=True)
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init


if __name__ == '__main__':
    vec_env = SubprocVecEnv([make_env(i) for i in range(1)])
    vec_env = VecFrameStack(vec_env, n_stack=4)

    model = PPO.load('best_model', env=vec_env)
    obs = vec_env.reset()

    for _ in range(int(1e4)):
        action, _state = model.predict(obs)
        obs, reward, done, info = vec_env.step(action)
        if reward != 0:
            print(reward)

