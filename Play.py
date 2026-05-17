"""
Play.py - Interactive human play mode for Zelda RL environment.
Controls:
  Arrow Keys  → Move Link
  Z           → A button (attack / interact)
  X           → B button
  Enter       → Start
  Q           → Quit

Usage:
  python Play.py
  python Play.py --state saved.state
"""
import argparse
import time
import pyboy
from pyboy.utils import WindowEvent
from Rom.Zelda_env import ZeldaEnv


KEYMAP = {
    "up":    WindowEvent.PRESS_ARROW_UP,
    "down":  WindowEvent.PRESS_ARROW_DOWN,
    "left":  WindowEvent.PRESS_ARROW_LEFT,
    "right": WindowEvent.PRESS_ARROW_RIGHT,
    "z":     WindowEvent.PRESS_BUTTON_A,
    "x":     WindowEvent.PRESS_BUTTON_B,
    "enter": WindowEvent.PRESS_BUTTON_START,
}
RELEASE_MAP = {
    "up":    WindowEvent.RELEASE_ARROW_UP,
    "down":  WindowEvent.RELEASE_ARROW_DOWN,
    "left":  WindowEvent.RELEASE_ARROW_LEFT,
    "right": WindowEvent.RELEASE_ARROW_RIGHT,
    "z":     WindowEvent.RELEASE_BUTTON_A,
    "x":     WindowEvent.RELEASE_BUTTON_B,
    "enter": WindowEvent.RELEASE_BUTTON_START,
}


def print_hud(env):
    """Print a live HUD to the terminal."""
    info = env._get_info()
    health = info.get("health", "?")
    sword  = "✓" if info.get("sword_level", 0) > 0 else "✗"
    shield = "✓" if info.get("shield_level", 0) > 0 else "✗"
    world  = info.get("current_world", "?")
    room   = info.get("overworld_room", "?")
    x      = info.get("player_x", "?")
    y      = info.get("player_y", "?")
    kills  = info.get("killed_monster", "?")
    visited = len(env.visited_location)
    lifetime = len(env.lifetime_visited)
    print(
        f"\r[HP:{health:>2}] [World:{world}] [Room:{room:>3}] [X:{x:>3} Y:{y:>3}] "
        f"[Sword:{sword}] [Shield:{shield}] "
        f"[Kills:{kills:>3}] [Tiles:{visited:>4}] [Lifetime:{lifetime:>4}]",
        end="", flush=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play Zelda manually in the RL environment.")
    parser.add_argument("--state", default="init.state", help="Save state to load (default: init.state)")
    parser.add_argument("--speed", type=int, default=1, help="Emulation speed (default: 1 = normal speed)")
    args = parser.parse_args()

    # Use the raw ZeldaEnv but with show=True so the SDL2 window appears
    env = ZeldaEnv(pos=99, save=False, show=True, speed=args.speed)
    env.init_state = args.state
    env.reset()

    print(f"\n[PLAY MODE] Loaded state: {args.state}")
    print("Controls: Arrow Keys to move, Z=A, X=B, Enter=Start, Q=Quit")
    print("─" * 70)

    try:
        # Use pynput for non-blocking keyboard reading
        try:
            from pynput import keyboard as kb
        except ImportError:
            print("\n[ERROR] pynput is required for keyboard input.")
            print("Install it with: pip install pynput")
            env.close()
            exit(1)

        pressed = set()

        def on_press(key):
            try:
                k = key.char.lower() if hasattr(key, "char") and key.char else key.name.lower()
            except AttributeError:
                return
            if k == "q":
                return False  # Stop listener
            if k not in pressed:
                pressed.add(k)
                event = KEYMAP.get(k)
                if event:
                    env.pyboy.send_input(event)

        def on_release(key):
            try:
                k = key.char.lower() if hasattr(key, "char") and key.char else key.name.lower()
            except AttributeError:
                return
            pressed.discard(k)
            event = RELEASE_MAP.get(k)
            if event:
                env.pyboy.send_input(event)

        listener = kb.Listener(on_press=on_press, on_release=on_release)
        listener.start()

        print("[INFO] Window is open. Press Q to quit.\n")

        while listener.is_alive():
            env.pyboy.tick(render=True)
            print_hud(env)
            time.sleep(1 / 60)  # Cap at 60fps for HUD refresh

    except KeyboardInterrupt:
        pass
    finally:
        print("\n[PLAY MODE] Exiting...")
        env.close()
