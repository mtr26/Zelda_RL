from Rom.memory_adress import PLAYER_X
from typing import Any
import time
import os
import gymnasium as gym
from pyboy.utils import WindowEvent
import pyboy
import numpy as np
from Rom.memory_adress import *
from Rom.memory_adress import *
import json
from pathlib import Path
import mediapy as media


with open('Rom/reward_table.json', 'r+') as f:
    reward_table = json.load(f)

PATH = 'Rom/bin/zelda.gbc'

# TODO : Implement a negatif reward system when the model return in an already known area

class ZeldaEnv(gym.Env):
    def __init__(self, pos, show = False, save = True, speed = 0, max_step = 2048*8,
                 curriculum_states = None, curriculum_weights = None, act_freq = 24):
        super().__init__()
        self.act_freq = act_freq
        self.init_state = 'init.state'
        # Curriculum: list of .state paths to sample from at reset.
        # curriculum_weights controls sampling probability (must sum to 1).
        self.curriculum_states = curriculum_states  # e.g. ['init.state', 'saved.state']
        self.curriculum_weights = curriculum_weights  # e.g. [0.7, 0.3]
        self.frame_stacks = 3
        self.step_count = 0
        self.max_step = max_step
        self.coverage_size = 64
        self.coverage_by_world = {}
        self.coverage_dirty = False
        self.event_r = 0.0
        self.last_reward = 0.0
        self.last_reward_components = {
            "explore": 0.0,
            "fight": 0.0,
            "event": 0.0,
            "coverage": 0.0,
            "kill": 0.0,
            "stuck": 0.0,
            "step": 0.0,
            "death": 0.0,
        }
        self.coverage_r = 0.0
        self.kill_r = 0.0
        self.stuck_r = 0.0
        self.step_r = 0.0
        self.death_r = 0.0
        self.last_coverage_new = False
        self.stuck_threshold = 64
        self.stuck_steps = 0
        self.last_position = None
        self.last_info = None
        self.episode_start_time = time.time()

        self.got_shield = False
        self.got_sword = False
        self.visited_location = set()
        self.visited_worlds = set()
        self.kill_count = 0
        self.lifetime_visited = set()  # persists across episodes; only new tiles earn explore reward
        self.prev_lifetime_count = 0   # snapshot of lifetime_visited size at last step (delta-based reward)
        self._episode_start_kills = 0
        self._start_state_used = 'init.state'
        self.last_episode_summary = {}
        # Per-episode accumulators — must be initialized here so clear() can read them on ep0
        self.reward_sum = 0.0
        self.explo = 0.0
        self.heal_r = 0.0


        self.metadata = {"render.modes": []}
        self.reward_range = (0, 15000)

        self.release_arrow = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP
        ]

        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            None,
        ]

        self.release_button = [
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B
        ]

        self.release_actions = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
        ]


        self.output_shape = (36, 40, 1)
        self.mem_padding = 2
        self.memory_height = 8
        self.col_steps = 16
        self.output_full = (
            self.output_shape[0] * self.frame_stacks + 2 * (self.mem_padding + self.memory_height),
                            self.output_shape[1],
                            self.output_shape[2]
        )

        # Set these in ALL subclasses
        self.action_space = gym.spaces.Discrete(len(self.valid_actions))
        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(low=0, high=255, shape=self.output_shape, dtype=np.uint8),
            "ram": gym.spaces.Box(low=0.0, high=1.0, shape=(25,), dtype=np.float32)
        })

        self.statut = False
        self.pyboy = pyboy.PyBoy(
            PATH,
            window='SDL2' if show else 'null',
            sound_emulated=False,
                )
        self.show = show
        self.screen = self.pyboy.screen
        self.old_info = dict(self._get_info())
        # Match Pokemon Red: run at full speed when headless, throttle to 6x when viewing
        self.pyboy.set_emulation_speed(6 if show else 0)
        self.reset_number = 0
        self.reset()
        self.save = save
        if save:
            self.initialize(pos=pos)

    """
    def init_knn(self):
        # Declaring index
        self.knn_index = hnswlib.Index(space='l2', dim=self.vec_dim) # possible options are l2, cosine or ip
        # Initing index - the maximum number of elements should be known beforehand
        self.knn_index.init_index(
            max_elements=self.num_elements, ef_construction=100, M=16)
    """
    def initialize(self, pos):
        if not self.save:
            return
        os.makedirs('vid', exist_ok=True)
        self.full_frame_writer = media.VideoWriter(f'vid/full_{pos}.mp4', (144, 160), fps=60)
        self.full_frame_writer.__enter__()

    def _choose_start_state(self):
        """Return a state file path, sampling from curriculum if configured."""
        if self.curriculum_states and len(self.curriculum_states) > 0:
            chosen = np.random.choice(self.curriculum_states, p=self.curriculum_weights)
        else:
            chosen = self.init_state
        self._start_state_used = chosen
        return chosen

    def reset(self, seed = None, options = None):
        # TODO : implement the reset method
        self.seed = seed
        # Curriculum: sample a starting state
        state_path = self._choose_start_state()
        with open(state_path, "rb") as f:
            self.pyboy.load_state(f)
        

        self.recent_memory = np.zeros((self.output_shape[1]*self.memory_height, 3), dtype=np.uint8)

        
        
        self.recent_frames = np.zeros(
            (self.frame_stacks, self.output_shape[0], 
             self.output_shape[1], self.output_shape[2]),
            dtype=np.uint8)
        
        self.step_count = 0
        self.statut = False
        self.old_info = dict(self._get_info())
        self.clear()
        self.episode_start_time = time.time()
        self.last_info = None
        self.reset_number += 1
        
        info = self._get_info()
        obs = {"image": self.render(), "ram": self._get_ram_state(info)}
        return obs, info
    
    
    def render(self, is_resize = True):
        # The raw PyBoy screen is shape (144, 160, 4) (RGBA), we just take RGB
        game_pixels_render = self.screen.ndarray[..., :3]
        if is_resize:
            # Downscale by 4x using numpy slicing (O(1) fast vs skimage)
            small = game_pixels_render[::4, ::4]
            # Convert to Grayscale using ITU-R 601 Luma
            return np.dot(small, [0.299, 0.587, 0.114]).astype(np.uint8)[..., np.newaxis]
        return game_pixels_render
    
    def get_image(self):
        return self.render(is_resize=False)
    
    def add_video_frame(self):
        self.full_frame_writer.add_image(self.get_image())
    
    def step(self, action: Any):
        # TODO : implement the step method
        self._action_on_emulator(action)
        info = self._get_info()
        obs = {"image": self.render(), "ram": self._get_ram_state(info)}
        self.last_info = info
        self._update_coverage(info)
        self._update_stuck(info)
        rewards = self._get_rewards(info) * 0.1
        terminated = info.get("health", 1) <= 0
        truncated = self.step_count >= self.max_step
        if not truncated:
            self.step_count += 1
        self.statut = terminated or truncated
        return obs, rewards, terminated, truncated, info
    
    def close(self):
        self.pyboy.stop()
        if self.save:
            self.full_frame_writer.__exit__(None, None, None)


    def check_change(self):
        return {i:self.old_info[i]!=self._get_info()[i] for i in self.old_info.keys()}
    
    def update(self, info = None):
        base_info = info if info is not None else self._get_info()
        self.old_info = dict(base_info)

    def clear(self):
        # Snapshot completed-episode stats BEFORE resetting (readable via get_episode_summary)
        kills_this_ep = max(0, self.kill_count - self._episode_start_kills)
        self.last_episode_summary = {
            'reward_sum': float(self.reward_sum),
            'reward_explore': float(self.explo),
            'reward_event': float(self.event_r),
            'reward_kill': float(self.kill_r),
            'reward_fight': float(self.heal_r),
            'reward_stuck': float(self.stuck_r),
            'reward_coverage': float(self.coverage_r),
            'reward_step': float(self.step_r),
            'reward_death': float(self.death_r),
            'has_shield': bool(self.got_shield),
            'has_sword': bool(self.got_sword),
            'kills_this_episode': int(kills_this_ep),
            'visited_locations': int(len(self.visited_location)),
            'lifetime_visited': int(len(self.lifetime_visited)),
            'stuck_steps_final': int(self.stuck_steps),
            'episode_length': int(self.step_count),
            'world_at_end': int(self._get_current_world),
            'start_state': str(self._start_state_used),
        }

        self.got_sword = False
        self.got_shield = False
        self.explo = 0
        self.heal_r = 0
        self.event_r = 0
        self.reward_sum = 0
        self.last_reward = 0.0
        self.last_reward_components = {
            "explore": 0.0,
            "fight": 0.0,
            "event": 0.0,
            "coverage": 0.0,
            "kill": 0.0,
            "stuck": 0.0,
            "step": 0.0,
            "death": 0.0,
        }
        self.coverage_r = 0.0
        self.kill_r = 0.0
        self.stuck_r = 0.0
        self.step_r = 0.0
        self.death_r = 0.0
        self.last_coverage_new = False
        self.stuck_steps = 0
        self._episode_start_kills = int(self._get_nbr_killed_monster)
        self.kill_count = self._episode_start_kills
        # Reset exploration tracking for this episode.
        # Per-episode reset (like Pokemon RL's KNN) ensures every episode
        # starts on equal footing so SaveOnBestCallback works correctly.
        self.lifetime_visited.clear()
        self.prev_lifetime_count = 0
        self.visited_location = set()
        self.visited_location.add((
            self._get_current_world,
            self._get_overworld_room,
            self._get_player_x // 10,
            self._get_player_y // 10,
        ))
        self.visited_worlds = {self._get_current_world}
        self.locations = self._get_maps_statuts
        self.life = self._get_health_level
        self.coverage_by_world = {}
        self.coverage_dirty = False
        self.last_position = (self._get_player_x, self._get_player_y)
        self._update_coverage_from_position(self._get_current_world, self._get_player_x, self._get_player_y)


    def _reset_memory(self):
        self.pyboy.memory[WORLD_STATUT[0]:WORLD_STATUT[1]] = [0]*len(self.pyboy.memory[WORLD_STATUT[0]:WORLD_STATUT[1]])
    
    def _skip_frame(self):
        self.pyboy.tick(count=25, render=False)

    def exploration_reward(self, change, info):
        """Delta-based exploration reward (Pokemon-style).
        
        We add every visited tile to lifetime_visited (which never resets).
        The reward each step = (new lifetime_visited size - prev size) * per_tile_value.
        This means:
          - Pacing between two known rooms → delta = 0 → reward = 0
          - Walking into a brand new tile → delta = 1 → reward = per_tile_value
          - No farming possible: a tile only contributes to the delta once, ever.
        """
        loc = info['player_location']
        # Always track within-episode visits (used by stuck detection)
        self.visited_location.add(loc)
        # Add to lifetime set; it will only grow when loc is truly new
        self.lifetime_visited.add(loc)

        # Compute delta since last step
        new_count = len(self.lifetime_visited)
        delta = new_count - self.prev_lifetime_count
        self.prev_lifetime_count = new_count

        per_tile = reward_table.get('explore_per_tile', 0.005)
        reward = delta * per_tile
        self.explo += reward
        return reward

    def fight_reward(self, change, info):
        reward = 0
        if change['health'] and self.life > info['health']:
            reward += reward_table['hit']
            self.life = self._get_info()['health']

        elif change['health'] and self.life < info['health']:
            reward += reward_table['more_life']
            self.life = info['health']
        self.heal_r += reward
        return reward

    def event_reward(self, change):
        reward = 0
        if change['shield_level'] and self.got_shield == False:
            reward += reward_table['shield']
            self.got_shield = True
        
        if change['sword_level'] and self.got_sword == False:
            self.got_sword = True
            reward += reward_table['sword']
        self.event_r += reward
        return reward

    def kill_reward(self, info):
        """Reward each new monster kill since episode start."""
        current_kills = int(info.get('killed_monster', 0))
        new_kills = max(0, current_kills - self.kill_count)
        self.kill_count = max(self.kill_count, current_kills)
        reward = new_kills * reward_table.get('kill', 0.0)
        self.kill_r += reward
        return reward

    def print_reward(self):
        txt = f'step: {self.step_count}  shield: {self.got_shield}  sword: {self.got_sword}  kills: {self.kill_count}  explo: {self.explo}  heal: {self.heal_r}  sum: {self.reward_sum}'
        if self.show:
            print(f'\r{txt}', end='', flush=True)
        else:
            with open('output.txt', 'w+') as f:
                print(f'\r{txt}', end='', flush=True, file=f)
    
    def _get_rewards(self, info):
        '''The reward method'''
        changes = {key: self.old_info.get(key) != info.get(key) for key in info.keys()}
        if changes['map_statut']:
            self._skip_frame()
        reward_event = self.event_reward(changes)
        reward_explore = self.exploration_reward(changes, info)
        reward_fight = self.fight_reward(changes, info)
        reward_kill = self.kill_reward(info)
        reward_coverage = reward_table.get("coverage", 0.0) if self.last_coverage_new else 0.0
        reward_stuck = 0.0
        if self.stuck_steps >= self.stuck_threshold:
            reward_stuck = reward_table.get("stuck", 0.0)
            
        reward_step = reward_table.get("step", 0.0)
        reward_death = 0.0
        if info.get("health", 1) <= 0:
            reward_death = reward_table.get("loss", 0.0)
            
        reward = reward_event + reward_explore + reward_fight + reward_kill + reward_coverage + reward_stuck + reward_step + reward_death
        self.last_reward_components = {
            "explore": reward_explore,
            "fight": reward_fight,
            "event": reward_event,
            "kill": reward_kill,
            "coverage": reward_coverage,
            "stuck": reward_stuck,
            "step": reward_step,
            "death": reward_death,
        }
        self.last_reward = reward
        self.coverage_r += reward_coverage
        self.stuck_r += reward_stuck
        self.step_r += reward_step
        self.death_r += reward_death
        self.reward_sum += reward
        self.print_reward()
        # NOTE: _reset_memory() was intentionally removed — it was zeroing
        # WORLD_STATUT (0xD800-0xD8FF) every step, wiping the game's room-visited
        # flags and causing map_discovered to always read 0 in debug stats.
        self.update(info)
        return reward
    
    def _action_on_emulator(self, action):
        event = self.valid_actions[action]
        headless = not self.save and not self.show
        if event is None:
            # No-op action: skip render on intermediate steps in headless mode
            for i in range(self.act_freq):
                is_last = (i == self.act_freq - 1)
                if self.save:
                    self.add_video_frame()
                self.pyboy.tick(render=not headless or is_last)
            return
        self.pyboy.send_input(event)
        for i in range(self.act_freq):
            is_last = (i == self.act_freq - 1)
            if i == 8:
                if action < 4:
                    self.pyboy.send_input(self.release_arrow[action])
                if action > 3 and action < 6:
                    self.pyboy.send_input(self.release_button[action - 4])
                if event == WindowEvent.PRESS_BUTTON_START:
                    self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
            if self.save:
                self.add_video_frame()
            # Only render on the last frame; skip pixel buffer on intermediates
            self.pyboy.tick(render=not headless or is_last)
        #self.pyboy.send_input(self.release_actions[action])

    def _update_coverage_from_position(self, world, x, y):
        world_id = int(world)
        grid = self.coverage_by_world.get(world_id)
        if grid is None:
            grid = np.zeros((self.coverage_size, self.coverage_size), dtype=np.uint16)
            self.coverage_by_world[world_id] = grid
        scale = 256
        grid_x = min(self.coverage_size - 1, max(0, x * self.coverage_size // scale))
        grid_y = min(self.coverage_size - 1, max(0, y * self.coverage_size // scale))
        was_new = grid[grid_y, grid_x] == 0
        if grid[grid_y, grid_x] < np.iinfo(grid.dtype).max:
            grid[grid_y, grid_x] += 1
        self.last_coverage_new = was_new
        self.coverage_dirty = True

    def _update_coverage(self, info):
        self._update_coverage_from_position(info["current_world"], info["player_x"], info["player_y"])

    def _update_stuck(self, info):
        pos = info["player_location"]
        if pos == self.last_position:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0
            self.last_position = pos

    def get_coverage_map(self, world = None):
        world_id = int(self._get_current_world if world is None else world)
        grid = self.coverage_by_world.get(world_id)
        if grid is None:
            return np.zeros((self.coverage_size, self.coverage_size), dtype=np.uint8)
        return np.clip(grid, 0, 255).astype(np.uint8)

    def get_coverage_stats(self, world = None):
        coverage = self.get_coverage_map(world)
        visited = int(np.count_nonzero(coverage))
        total = int(coverage.size)
        ratio = visited / total if total else 0.0
        return {"coverage_visited": visited, "coverage_total": total, "coverage_ratio": ratio}

    def get_ascii_map(self, world = None, size = 32):
        coverage = self.get_coverage_map(world)
        if size > 0 and size != coverage.shape[0] and coverage.shape[0] % size == 0:
            factor = coverage.shape[0] // size
            coverage = coverage.reshape(size, factor, size, factor).sum(axis=(1, 3))
        coverage = coverage[:size, :size]
        lines = []
        for row in coverage:
            lines.append("".join("#" if val > 0 else "." for val in row))
        return "\n".join(lines)

    def get_episode_summary(self):
        """Return the summary of the most recently completed episode.
        Safe to call right after reset() — snapshot is taken in clear() before any reset.
        """
        return dict(self.last_episode_summary)

    def get_debug_stats(self):
        info = self.last_info if self.last_info is not None else self._get_info()
        map_statut = info["map_statut"]
        discovered = int(np.count_nonzero(np.array(map_statut)))
        total = int(len(map_statut))
        ratio = discovered / total if total else 0.0
        coverage_stats = self.get_coverage_stats(info["current_world"])
        # Reward component percentages (for farming detection)
        total_r = self.reward_sum if self.reward_sum != 0 else 1e-9
        return {
            "step": int(self.step_count),
            "timesteps": int(self.step_count),
            "world": int(info["current_world"]),
            "player_x": int(info["player_x"]),
            "player_y": int(info["player_y"]),
            "health": int(info["health"]),
            "killed_monster": int(self.kill_count),
            "reward_sum": float(self.reward_sum),
            "reward_explore": float(self.explo),
            "reward_fight": float(self.heal_r),
            "reward_event": float(self.event_r),
            "reward_kill": float(self.kill_r),
            "reward_coverage": float(self.coverage_r),
            "reward_stuck": float(self.stuck_r),
            "reward_step": float(self.step_r),
            "reward_death": float(self.death_r),
            "explore_pct": float(self.explo / total_r),
            "event_pct": float(self.event_r / total_r),
            "kill_pct": float(self.kill_r / total_r),
            "last_reward": float(self.last_reward),
            "last_reward_components": dict(self.last_reward_components),
            "map_discovered": discovered,
            "map_total": total,
            "map_coverage_ratio": float(ratio),
            "coverage_visited": coverage_stats["coverage_visited"],
            "coverage_total": coverage_stats["coverage_total"],
            "coverage_ratio": float(coverage_stats["coverage_ratio"]),
            "visited_locations": int(len(self.visited_location)),
            "lifetime_visited": int(len(self.lifetime_visited)),
            "visited_worlds": int(len(self.visited_worlds)),
            "stuck_steps": int(self.stuck_steps),
            "has_sword": bool(self.got_sword),
            "has_shield": bool(self.got_shield),
            "start_state": str(self._start_state_used),
            "episode_seconds": float(time.time() - self.episode_start_time),
        }
    
    

    # MARK: memory readers
    
    @property
    def _get_current_world(self):
        '''Get the current world (to determine)'''
        return self.pyboy.memory[CURRENT_WORLD]
    
    @property
    def _get_player_x(self):
        '''Get the player's in-room X pixel position (0x00–0xF0)'''
        return self.pyboy.memory[PLAYER_X_INROOM]

    @property
    def _get_player_y(self):
        '''Get the player's in-room Y pixel position (0x3D–0xDD)'''
        return self.pyboy.memory[PLAYER_Y_INROOM]
    
    @property
    def _get_overworld_room(self):
        '''Get the current map/room ID (0xFFF6). Changes when Link transitions between screens.'''
        return self.pyboy.memory[MAP_ROOM_ID]

    @property
    def _get_dungeon(self):
        '''Get the current dungeon/sub-map number (0xD402). 0xFF = Color Dungeon.'''
        return self.pyboy.memory[CURRENT_DUNGEON]
    
    @property
    def _get_maps_statuts(self):
        '''Get maps statuts'''
        return self.pyboy.memory[WORLD_STATUT[0]:WORLD_STATUT[1]]
    
    @property
    def _get_held_items(self):
        '''Get both hield items'''
        return self.pyboy.memory[HELD_ITEM_1:HELD_ITEM_2]
    
    
    @property
    def _get_player_inventory(self):
        '''Get the player's inventory'''
        return self.pyboy.memory[INVENTORY[0]:INVENTORY[1]]
    
    @property
    def _get_number_bombs(self):
        '''Get the player's numbers of bombs'''
        return self.pyboy.memory[NUMBER_BOMBS]
    
    @property
    def _get_number_arrows(self):
        '''Get the player's numbers of arrows'''
        return self.pyboy.memory[NUMBER_ARROWS]
    
    @property
    def _get_shield_level(self):
        '''Get the shield statut'''
        return self.pyboy.memory[SHIELD_LEVEL]
    
    @property
    def _get_sword_level(self):
        '''Get the sword statut'''
        return self.pyboy.memory[SWORD_LEVEL]
    
    @property
    def _get_health_level(self):
        '''Refer to the memory_adress file'''
        return self.pyboy.memory[CURRENT_HEALTH]
    
    @property
    def _get_max_bombs(self):
        '''Get the maximum number of bombs'''
        return self.pyboy.memory[MAX_BOMB]
    
    @property
    def _get_max_arrows(self):
        '''Get the maximum number of arrows'''
        return self.pyboy.memory[MAX_ARROWS]
    
    @property
    def _get_nbr_killed_monster(self):
        '''Get the number of killed monsters'''
        return self.pyboy.memory[KILLED_MONSTERS[0]]
    
    @property
    def _get_items_list(self):
        '''Refer to the items table'''
        item_list = {'01' : False, '02' : False, '03' : False, '04' : False, '05' : False, '06' : False, '07' : False, '08' : False, '09' : False, '0A' : False, '0B' : False, '0C' : False, '0D' : False}
        for item_slot in self._get_player_inventory:
            if item_slot in item_list.keys():
                item_list[item_slot] = True
        return item_list

    def _get_ram_state(self, info):
        # in-room pixel X (0-160)
        px = info.get('player_x', 0) / 160.0
        # in-room pixel Y (0-144)
        py = info.get('player_y', 0) / 144.0
        # area type: 0=overworld, 1=dungeon/indoor, 2=side-scroll → normalize to [0,1]
        world = info.get('current_world', 0) / 2.0
        # map room ID (0-255)
        room = info.get('overworld_room', 0) / 255.0
        health = info.get('health', 0) / 24.0
        kills = info.get('killed_monster', 0) / 255.0
        
        held_items = info.get('held_items', [])
        h1 = 0.0
        if len(held_items) > 0 and isinstance(held_items[0], str):
            try: h1 = float(int(held_items[0], 16)) / 13.0
            except: pass
        h2 = 0.0
        if len(held_items) > 1 and isinstance(held_items[1], str):
            try: h2 = float(int(held_items[1], 16)) / 13.0
            except: pass
            
        sword_lvl = info.get('sword_level', 0) / 2.0
        shield_lvl = info.get('shield_level', 0) / 2.0
        
        mb = info.get('max_bombs', 1)
        if mb == 0: mb = 1
        ma = info.get('maw_arrows', 1) # Note: typo in info 'maw_arrows' matches _get_info output
        if ma == 0: ma = 1
        bombs = info.get('nbr_bombs', 0) / float(mb)
        arrows = info.get('nbr_arrows', 0) / float(ma)
        
        item_list = info.get('item_list', {})
        inv_flags = []
        for i in range(1, 14):
            key = f'{i:02X}'
            inv_flags.append(1.0 if item_list.get(key, False) else 0.0)
            
        ram = [px, py, world, room, health, kills, h1, h2, sword_lvl, shield_lvl, bombs, arrows] + inv_flags
        return np.array(ram, dtype=np.float32)

    def _get_info(self):
        # TODO : add the bolean gets
        '''Return the info after a step occurs'''
        return dict(
            current_world=self._get_current_world,
            player_x=self._get_player_x,
            player_y=self._get_player_y,
            overworld_room=self._get_overworld_room,
            dungeon=self._get_dungeon,
            # Canonical exploration key: (area_type, overworld_room, coarse_x, coarse_y)
            # area_type 0=overworld, 1=dungeon/indoor, 2=side-scroll
            # coarse position divides 160-px room into 10-px tiles (16 buckets per axis)
            player_location=(
                self._get_current_world,
                self._get_overworld_room,
                self._get_player_x // 10,
                self._get_player_y // 10,
            ),
            map_statut=self._get_maps_statuts,
            held_items=self._get_held_items,
            inventory=self._get_player_inventory,
            nbr_bombs=self._get_number_bombs,
            nbr_arrows=self._get_number_arrows,
            shield_level=self._get_shield_level,
            sword_level=self._get_sword_level,
            health=self._get_health_level,
            max_bombs=self._get_max_bombs,
            maw_arrows=self._get_max_arrows,
            item_list=self._get_items_list,
            killed_monster=self._get_nbr_killed_monster,
            visited_location=self.visited_location,
            visited_world=self.visited_worlds
        )
    
    
    

    

    

