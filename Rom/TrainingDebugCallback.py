import csv
import json
import os
from collections import deque

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3.common.callbacks import BaseCallback


class TrainingDebugCallback(BaseCallback):
    """
    Callback that:
    - Logs per-step debug stats to debug_stats.jsonl (for audit_run.py / smart_audit.py)
    - Logs per-episode stats to episode_stats.csv when any env finishes an episode
    - Records rolling shield/sword acquisition rates to TensorBoard
    - Saves coverage images and ASCII maps at configurable intervals
    """
    def __init__(
        self,
        check_freq: int,
        log_dir: str,
        coverage_every: int = 0,
        text_map_every: int = 0,
        rolling_window: int = 100,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.coverage_every = coverage_every
        self.text_map_every = text_map_every
        self.rolling_window = rolling_window

        self.debug_stats_path = os.path.join(log_dir, "debug_stats.jsonl")
        self.episode_csv_path = os.path.join(log_dir, "episode_stats.csv")
        self.coverage_dir = os.path.join(log_dir, "coverage")
        self.text_map_path = os.path.join(log_dir, "coverage.txt")

        self._last_stats = None
        self._episode_csv_initialized = False
        self._episode_number = 0

        # Rolling windows for rate tracking
        self._shield_window = deque(maxlen=rolling_window)
        self._sword_window = deque(maxlen=rolling_window)
        self._reward_window = deque(maxlen=rolling_window)
        self._explore_pct_window = deque(maxlen=rolling_window)

    def _init_callback(self) -> None:
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.coverage_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Episode CSV helpers
    # ------------------------------------------------------------------
    _EPISODE_CSV_FIELDS = [
        "episode", "timestep", "start_state", "episode_length",
        "reward_sum", "reward_explore", "reward_event", "reward_kill",
        "reward_fight", "reward_stuck", "reward_coverage", 
        "reward_step", "reward_death",
        "has_shield", "has_sword", "kills_this_episode",
        "visited_locations", "lifetime_visited",
        "stuck_steps_final", "world_at_end",
    ]

    def _ensure_episode_csv(self):
        if self._episode_csv_initialized:
            return
        write_header = not os.path.exists(self.episode_csv_path)
        self._ep_csv_file = open(self.episode_csv_path, "a", newline="", encoding="utf-8")
        self._ep_csv_writer = csv.DictWriter(
            self._ep_csv_file, fieldnames=self._EPISODE_CSV_FIELDS, extrasaction="ignore"
        )
        if write_header:
            self._ep_csv_writer.writeheader()
        self._episode_csv_initialized = True

    def _log_episode_summary(self, env_idx: int):
        """Called when env_idx finishes an episode. Queries env for the completed episode summary."""
        try:
            summaries = self.training_env.env_method("get_episode_summary", indices=[env_idx])
        except Exception:
            return
        if not summaries or not summaries[0]:
            return
        summary = summaries[0]
        self._episode_number += 1
        summary["episode"] = self._episode_number
        summary["timestep"] = int(self.num_timesteps)

        # Write to CSV
        self._ensure_episode_csv()
        self._ep_csv_writer.writerow(summary)
        self._ep_csv_file.flush()

        # Update rolling windows
        self._shield_window.append(1 if summary.get("has_shield") else 0)
        self._sword_window.append(1 if summary.get("has_sword") else 0)
        self._reward_window.append(summary.get("reward_sum", 0.0))
        # explore_pct not in summary but we can compute it
        r = summary.get("reward_sum", 0) or 1e-9
        self._explore_pct_window.append(summary.get("reward_explore", 0) / r)

    # ------------------------------------------------------------------
    # Main step callback
    # ------------------------------------------------------------------
    def _on_step(self) -> bool:
        # Check for episode ends across all envs
        dones = self.locals.get("dones", [])
        for i, done in enumerate(dones):
            if done:
                self._log_episode_summary(i)

        # Periodic debug stats to jsonl + TensorBoard
        if self.check_freq > 0 and self.n_calls % self.check_freq == 0:
            stats = self._get_debug_stats()
            if stats is not None:
                self._last_stats = stats
                self._record_stats(stats)
                self._append_stats(stats)

        # Periodic coverage image
        if self.coverage_every > 0 and self.n_calls % self.coverage_every == 0:
            self._save_coverage_image()

        # Periodic ASCII text map
        if self.text_map_every > 0 and self.n_calls % self.text_map_every == 0:
            self._save_text_map()

        return True

    # ------------------------------------------------------------------
    # Stats collection
    # ------------------------------------------------------------------
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
        # Per-step debug scalars
        self.logger.record("debug/coverage_ratio", stats.get("coverage_ratio", 0.0))
        self.logger.record("debug/map_coverage_ratio", stats.get("map_coverage_ratio", 0.0))
        self.logger.record("debug/visited_locations", stats.get("visited_locations", 0))
        self.logger.record("debug/lifetime_visited", stats.get("lifetime_visited", 0))
        self.logger.record("debug/visited_worlds", stats.get("visited_worlds", 0))
        self.logger.record("debug/stuck_steps", stats.get("stuck_steps", 0))
        self.logger.record("debug/health", stats.get("health", 0))
        self.logger.record("debug/killed_monster", stats.get("killed_monster", 0))

        # Reward breakdown
        self.logger.record("reward/episode_sum", stats.get("reward_sum", 0.0))
        self.logger.record("reward/explore_sum", stats.get("reward_explore", 0.0))
        self.logger.record("reward/fight_sum", stats.get("reward_fight", 0.0))
        self.logger.record("reward/event_sum", stats.get("reward_event", 0.0))
        self.logger.record("reward/kill_sum", stats.get("reward_kill", 0.0))
        self.logger.record("reward/coverage_sum", stats.get("reward_coverage", 0.0))
        self.logger.record("reward/stuck_sum", stats.get("reward_stuck", 0.0))
        self.logger.record("reward/step_sum", stats.get("reward_step", 0.0))
        self.logger.record("reward/death_sum", stats.get("reward_death", 0.0))
        self.logger.record("reward/last", stats.get("last_reward", 0.0))
        # Reward component percentages (farming detector)
        self.logger.record("reward/explore_pct", stats.get("explore_pct", 0.0))
        self.logger.record("reward/event_pct", stats.get("event_pct", 0.0))
        self.logger.record("reward/kill_pct", stats.get("kill_pct", 0.0))

        # Rolling episode rates (only meaningful once window fills)
        if self._shield_window:
            self.logger.record("rates/shield_rate", float(sum(self._shield_window) / len(self._shield_window)))
        if self._sword_window:
            self.logger.record("rates/sword_rate", float(sum(self._sword_window) / len(self._sword_window)))
        if self._reward_window:
            self.logger.record("rates/mean_episode_reward", float(sum(self._reward_window) / len(self._reward_window)))
        if self._explore_pct_window:
            self.logger.record("rates/mean_explore_pct", float(sum(self._explore_pct_window) / len(self._explore_pct_window)))

    def _append_stats(self, stats):
        with open(self.debug_stats_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(stats) + "\n")

    # ------------------------------------------------------------------
    # Coverage image / text map
    # ------------------------------------------------------------------
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
