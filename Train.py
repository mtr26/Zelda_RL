from Rom.Zelda_env import *
from stable_baselines3.common.vec_env import VecFrameStack, VecMonitor, SubprocVecEnv, DummyVecEnv, VecTransposeImage
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from Rom.SaveOnBestCallback import SaveOnBestTrainingRewardCallback
from Rom.TrainingDebugCallback import TrainingDebugCallback
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import plot_results
import os
import matplotlib

matplotlib.use("Agg")
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
    argparser.add_argument('--num_episodes', type=int, default=int(1e6))
    argparser.add_argument('--num_cpu', type=int, default=10)
    argparser.add_argument('--log_dir', type=str, default="tmp/")
    argparser.add_argument('--max_steps', type=int, default=2048 * 10)
    argparser.add_argument('--debug_freq', type=int, default=1000)
    argparser.add_argument('--coverage_freq', type=int, default=5000)
    argparser.add_argument('--text_map_freq', type=int, default=5000)
    argparser.add_argument('--checkpoint_freq', type=int, default=50000)
    argparser.add_argument('--ent_coef', type=float, default=0.01)
    argparser.add_argument('--pre_trained', action='store_true')
    argparser.add_argument('--show_plot', action='store_true')

    args = argparser.parse_args()
    num_episodes = args.num_episodes
    num_cpu = args.num_cpu
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    pre_trained = args.pre_trained

    end_model = True
    ep_length = args.max_steps

    vec_env = SubprocVecEnv([make_env(i, seed=0, max_time=ep_length) for i in range(num_cpu)])

    vec_env = VecMonitor(vec_env, log_dir)
    vec_env = VecFrameStack(vec_env, n_stack=4)
    vec_env = VecTransposeImage(vec_env)

    save_callback = SaveOnBestTrainingRewardCallback(check_freq=10, log_dir=log_dir)
    debug_dir = os.path.join(log_dir, "debug")
    debug_callback = TrainingDebugCallback(
        check_freq=args.debug_freq,
        log_dir=debug_dir,
        coverage_every=args.coverage_freq,
        text_map_every=args.text_map_freq,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix="ppo_zelda",
    )
    callback = CallbackList([save_callback, debug_callback, checkpoint_callback])

    
    if pre_trained:
        model = PPO.load('end_model' if end_model else 'best_model', env=vec_env)
        model.set_parameters('end_model' if end_model else 'best_model')
        model.rollout_buffer.buffer_size = ep_length
        model.rollout_buffer.reset()
    else:
        model = PPO(
            'CnnPolicy',
            env=vec_env,
            n_steps=ep_length // 8,
            batch_size=512,
            n_epochs=3,
            gamma=0.999,
            verbose=1,
            ent_coef=args.ent_coef,
            tensorboard_log=os.path.join(log_dir, "tb"),
        )

    total_timesteps = ep_length * num_cpu * num_episodes
    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=callback)
    model.save('end_model')

    try:
        x, y = results_plotter.ts2xy(results_plotter.load_results(log_dir), "timesteps")
        if len(x) > 0:
            plot_results([log_dir], total_timesteps, results_plotter.X_TIMESTEPS, "ZeldaTest")
            plt.savefig(os.path.join(log_dir, "training_curve.png"), dpi=150)
            if args.show_plot:
                plt.show()
        else:
            print("No monitor data available to plot yet.")
    except Exception as exc:
        print(f"Skipping plot: {exc}")
    
