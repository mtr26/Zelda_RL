import json
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3.common.callbacks import BaseCallback


class TrainingDebugCallback(BaseCallback):
    def __init__(
        self,
        check_freq: int,
        log_dir: str,
        coverage_every: int = 0,
        text_map_every: int = 0,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.coverage_every = coverage_every
        self.text_map_every = text_map_every
        self.debug_stats_path = os.path.join(log_dir, "debug_stats.jsonl")
        self.coverage_dir = os.path.join(log_dir, "coverage")
        self.text_map_path = os.path.join(log_dir, "coverage.txt")
        self._last_stats = None

    def _init_callback(self) -> None:
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.coverage_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.check_freq > 0 and self.n_calls % self.check_freq == 0:
            stats = self._get_debug_stats()
            if stats is not None:
                self._last_stats = stats
                self._record_stats(stats)
                self._append_stats(stats)

        if self.coverage_every > 0 and self.n_calls % self.coverage_every == 0:
            self._save_coverage_image()

        if self.text_map_every > 0 and self.n_calls % self.text_map_every == 0:
            self._save_text_map()

        return True

    def _get_debug_stats(self):
        try:
            stats_list = self.training_env.env_method("get_debug_stats", indices=[0])
        except Exception:
            return None
        if not stats_list:
            return None
        stats = stats_list[0]
        stats["timesteps"] = int(self.num_timesteps)
        return stats

    def _record_stats(self, stats):
        self.logger.record("debug/coverage_ratio", stats.get("coverage_ratio", 0.0))
        self.logger.record("debug/map_coverage_ratio", stats.get("map_coverage_ratio", 0.0))
        self.logger.record("debug/visited_locations", stats.get("visited_locations", 0))
        self.logger.record("debug/visited_worlds", stats.get("visited_worlds", 0))
        self.logger.record("debug/stuck_steps", stats.get("stuck_steps", 0))
        self.logger.record("debug/health", stats.get("health", 0))
        self.logger.record("debug/killed_monster", stats.get("killed_monster", 0))
        self.logger.record("reward/episode_sum", stats.get("reward_sum", 0.0))
        self.logger.record("reward/explore_sum", stats.get("reward_explore", 0.0))
        self.logger.record("reward/fight_sum", stats.get("reward_fight", 0.0))
        self.logger.record("reward/event_sum", stats.get("reward_event", 0.0))
        self.logger.record("reward/coverage_sum", stats.get("reward_coverage", 0.0))
        self.logger.record("reward/stuck_sum", stats.get("reward_stuck", 0.0))
        self.logger.record("reward/last", stats.get("last_reward", 0.0))

    def _append_stats(self, stats):
        with open(self.debug_stats_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(stats) + "\n")

    def _save_coverage_image(self):
        world = None
        if self._last_stats is not None:
            world = self._last_stats.get("world")
        try:
            coverage_list = self.training_env.env_method(
                "get_coverage_map", world=world, indices=[0]
            )
        except Exception:
            return
        if not coverage_list:
            return
        coverage = coverage_list[0]
        if coverage is None:
            return
        coverage = np.array(coverage, dtype=np.float32)
        filename = f"coverage_world_{world}_step_{self.num_timesteps}.png"
        out_path = os.path.join(self.coverage_dir, filename)
        plt.figure(figsize=(4, 4))
        plt.imshow(coverage, cmap="magma", interpolation="nearest")
        plt.axis("off")
        plt.tight_layout(pad=0.0)
        plt.savefig(out_path, dpi=150)
        plt.close()

    def _save_text_map(self):
        try:
            map_list = self.training_env.env_method(
                "get_ascii_map", size=32, indices=[0]
            )
        except Exception:
            return
        if not map_list:
            return
        text_map = map_list[0]
        with open(self.text_map_path, "w", encoding="utf-8") as f:
            f.write(f"timesteps={self.num_timesteps}\n")
            f.write(text_map)
            f.write("\n")
