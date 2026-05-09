import argparse
import queue
import select
import sys
import termios
import tty
import threading
from pathlib import Path

import pyboy as pb
from pyboy import PyBoy
import numpy as np

try:
    from pynput import keyboard as pynput_keyboard
except ImportError:
    pynput_keyboard = None

try:
    from Rom.memory_adress import *
except ImportError:
    from memory_adress import *


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
DEFAULT_ROM = BASE_DIR / 'bin' / 'zelda.gbc'
DEFAULT_STATE = REPO_ROOT / 'init.state'
DEFAULT_SAVE_STATE = REPO_ROOT / 'saved.state'


parser = argparse.ArgumentParser()
parser.add_argument('--rom', type=str, default=str(DEFAULT_ROM), help='Path to the Zelda ROM')
parser.add_argument('--state', type=str, default=str(DEFAULT_STATE), help='Path to a .state file to load')
parser.add_argument('--save-state', type=str, default=str(DEFAULT_SAVE_STATE), help='Path where the current emulator state will be saved when you press p')
args = parser.parse_args()

rom_path = Path(args.rom)
if not rom_path.is_absolute():
    rom_path = (REPO_ROOT / rom_path).resolve()

state_path = Path(args.state)
if not state_path.is_absolute():
    state_path = (REPO_ROOT / state_path).resolve()

save_state_path = Path(args.save_state)
if not save_state_path.is_absolute():
    save_state_path = (REPO_ROOT / save_state_path).resolve()

pyboy = PyBoy(str(rom_path))

control_queue = queue.Queue()


curent = None

def create():
    arr = []
    for i in range(0, 256):
        arr += pyboy.memory_scanner.scan_memory(i, start_addr=0x0000, end_addr=0xFFFF)
    return {i:pyboy.memory[i] for i in arr}

def choose_algo(dic, changes):
    for ele in dic.copy().keys():
        if pyboy.memory[ele] != dic[ele]:
            changes[ele] += 1
            dic[ele] = pyboy.memory[ele]
    return dic, changes

def choose_algo_2(dic):
    for ele in dic.copy().keys():
        if pyboy.memory[ele] != dic[ele]:
            del dic[ele]
    return dic

def choose_algo_3(dic, changes):
    for ele in dic.copy().keys():
        if pyboy.memory[ele] != dic[ele]:
            changes[ele] += 1
            dic[ele] = pyboy.memory[ele]
        else:
            del dic[ele]
    return dic, changes

def final(dic, change, i):
    for ele in dic.copy().keys():
        if change[ele] != i:
            del dic[ele]
    return dic

def save_change(dic, changes):
    with open('memory3.txt', 'w+') as f:
        print('''
---------------------------------------
|   index   |   values   |   change   |
|-----------|------------|------------|''', file=f)
        for ele in dic.keys():
            print(f'|   {hex(ele)}   |     {f"0{dic[ele]}" if dic[ele] < 10 else dic[ele]}     |     {f"0{changes[ele]}" if changes[ele] < 10 else changes[ele]}     |' ,file=f)

def print_game_data():
    """
    print({'world' : pyboy.memory[CURRENT_WORLD],
        'Player_pos' : (pyboy.memory[PLAYER_X], pyboy.memory[PLAYER_Y]),
        'Shield_level' : pyboy.memory[SHIELD_LEVEL],
        'Max_arrow' : pyboy.memory[MAX_ARROWS],
        'Current_health' : pyboy.memory[CURRENT_HEALTH],
        'Number_of_bomb' : pyboy.memory[NUMBER_BOMBS],
        'Number_of_arrow' : pyboy.memory[NUMBER_ARROWS],
        'Item_held' : (pyboy.memory[HELD_ITEM_1], pyboy.memory[HELD_ITEM_2])
              })"""
    
    #print(pyboy.memory_scanner.scan_memory(10, start_addr=0x0000, end_addr=0xFFFF))
    print(pyboy.memory[0xDBB5], pyboy.memory[0xF415])
    '''
    with open('Monster2.state', 'wb') as f:
        pyboy.save_state(f)
    '''


def save_game_state(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pyboy.save_state(f)
        f.flush()
    print(f'\nSaved state to: {path}')


def _key_to_action(key):
    if pynput_keyboard is None:
        return None

    if key in (pynput_keyboard.Key.esc,):
        return '__quit__'

    if key in (pynput_keyboard.Key.enter,):
        return 'start'

    if key in (pynput_keyboard.Key.space,):
        print(True)
        return '__save__'

    try:
        char = key.char.lower() if key.char else None
    except AttributeError:
        char = None

    if key in (pynput_keyboard.Key.up,) or char == 'w':
        return 'up'
    if key in (pynput_keyboard.Key.down,) or char == 's':
        return 'down'
    if key in (pynput_keyboard.Key.left,) or char == 'a':
        return 'left'
    if key in (pynput_keyboard.Key.right,) or char == 'd':
        return 'right'
    if char == 'z':
        return 'a'
    if char == 'x':
        return 'b'
    if char == 'p':
        return '__save__'
    if char == 'q':
        return '__quit__'

    return None


class KeyboardBridge:
    def __init__(self):
        self.pressed_actions = set()
        self.lock = threading.Lock()
        self.listener = None

    def on_press(self, key):
        action = _key_to_action(key)
        if action is None:
            return
        if action in ('__quit__', '__save__'):
            control_queue.put(('press', action))
            return
        with self.lock:
            if action in self.pressed_actions:
                return
            self.pressed_actions.add(action)
        control_queue.put(('press', action))

    def on_release(self, key):
        action = _key_to_action(key)
        if action is None or action in ('__quit__', '__save__'):
            return
        with self.lock:
            if action not in self.pressed_actions:
                return
            self.pressed_actions.remove(action)
        control_queue.put(('release', action))

    def start(self):
        if pynput_keyboard is None:
            return None
        self.listener = pynput_keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
        return self

    def stop(self):
        if self.listener is not None:
            self.listener.stop()


class TerminalInput:
    def __enter__(self):
        self.enabled = sys.stdin.isatty()
        self.old_settings = None
        if self.enabled:
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.enabled and self.old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def read_key(self):
        if not self.enabled:
            return None
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if not ready:
            return None
        char = sys.stdin.read(1)
        return char.lower()

# 0xDB55 - 0xDB9F : areas in the open world / related to the chunk
# 0xD900:0xDAFF - 0xDB3E:0xDB42 - 0xDB59 - 0xDB5F:0xDB60 - 0xDB6D:0xDB6E - 0xDB72:0xDB75 - 0xDB79:0xDB80 - 0xDB85:0xDB95 - 0xDBA0:0xDBA7 - 0xDBAB:0xDBAD : Nothing
# 0xDB4F:0xDB54 - 0xDB61:0xDB64 - 0xDB6F:0xDB71 - 0xDB80:0xDB84 - 0xDB96:0xDB9E - 0xDBA9:0xDAA - : IDK


# 0xd32e - 0xf32e - 0xe450 - 0xe120 - 0xe5ac - 0xc5ac - 0xc120 - 


def inp(key):
    curent = None
    if key in ('q', 'x', '\x1b'):
        return '__quit__'

    if key == 'j':
        return '__save__'

    if key in ('z', 'w'):
        pyboy.button_press('up')
        curent = 'up'

    elif key == 's':
        pyboy.button_press('down')
        curent = 'down'

    elif key == 'h':
        pyboy.button_press('left')
        curent = 'left'

    elif key == 'd':
        pyboy.button_press('right')
        curent = 'right'


    elif key == 'a':
        pyboy.button_press('a')
        curent = 'a'


    elif key == 'e':
        pyboy.button_press('b')
        curent = 'b'

    elif key in ('v', '\r', '\n'):
        pyboy.button_press('start')
        curent = 'start'

    if key == 'm':
        print_game_data()

    return curent


def _drain_control_queue():
    events = []
    while True:
        try:
            events.append(control_queue.get_nowait())
        except queue.Empty:
            break
    return events


if not state_path.exists():
    raise FileNotFoundError(f'Could not find state file: {state_path}')

with open(state_path, 'rb') as f:
    pyboy.load_state(f)

def get_levels_sum():
        poke_levels = [max(pyboy.memory[a] - 2, 0) for a in range(WORLD_STATUT[0], WORLD_STATUT[1]+1)]
        return max(sum(i > 0 for i in poke_levels) - 4, 0) 

def get_levels_reward(max_level_rew):
        explore_thresh = 22
        scale_factor = 4
        level_sum = get_levels_sum()
        if level_sum < explore_thresh:
            scaled = level_sum
        else:
            scaled = (level_sum-explore_thresh) / scale_factor + explore_thresh
        max_level_rew = max(max_level_rew, scaled)
        return max_level_rew


try:
    keyboard_bridge = KeyboardBridge().start() if pynput_keyboard is not None else None
    if keyboard_bridge is None:
        print('pynput is not available; falling back to terminal input. Use the terminal window for control.')

    #dic = create()
    #change = [0]*(2**16)
    if keyboard_bridge is not None:
        while True:
            quit_requested = False
            save_requested = False

            for kind, action in _drain_control_queue():
                if action == '__quit__':
                    quit_requested = True
                elif action == '__save__' and kind == 'press':
                    save_requested = True
                elif kind == 'press':
                    pyboy.button_press(action)
                elif kind == 'release':
                    pyboy.button_release(action)

            if quit_requested:
                break

            if save_requested:
                save_game_state(save_state_path)
                continue

            pyboy.tick()
            #dic, change = choose_algo(dic, change)
            #save_change(dic, change)
            print(pyboy.memory[0xff98], pyboy.memory[0xff99])# pyboy.memory[0xff98], pyboy.memory[0xffdb])
    else:
        with TerminalInput() as key_reader:
            while True:
                curent = inp(key_reader.read_key())
                if curent == '__quit__':
                    break
                if curent == '__save__':
                    save_game_state(save_state_path)
                    continue
                pyboy.tick()
                print(pyboy.memory[0xff98], pyboy.memory[0xff99])# pyboy.memory[0xff98], pyboy.memory[0xffdb])
                if curent != None:
                    pyboy.button_release(curent)
                    curent = None
finally:
    if 'keyboard_bridge' in locals() and keyboard_bridge is not None:
        keyboard_bridge.stop()
    pyboy.stop()


"""
dic = create()
change = [0]*(2**16)
for i in range(2):
    #pyboy.button_press('right')
    pyboy.tick()
    dic = choose_algo_2(dic)
    #save_change(dic, change)


pyboy.button_press('up')
pyboy.tick()
dic, change = choose_algo_3(dic, change)

pyboy.button_press('up')
pyboy.tick()
dic, change = choose_algo_3(dic, change)

pyboy.button_press('up')
pyboy.tick()
dic, change = choose_algo_3(dic, change)


dic = final(dic, change, 3)

save_change(dic, change)

"""





