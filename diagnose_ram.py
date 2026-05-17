"""
diagnose_ram.py — Walk Link around and watch candidate RAM addresses in real-time.
This will tell us EXACTLY which addresses track position correctly.

Usage:  python diagnose_ram.py
Controls: Arrow keys to move, Q to quit.
"""
import time
import pyboy
from pyboy.utils import WindowEvent

PATH = 'Rom/bin/zelda.gbc'

# All candidate addresses from documentation & disassembly
CANDIDATES = {
    # --- Position candidates ---
    "0x0070 (raw $70)":        0x0070,
    "0x0084 (raw $84)":        0x0084,
    "0xC070 (WRAM $70)":       0xC070,
    "0xC084 (WRAM $84)":       0xC084,
    "0xFF98 (sprite X)":       0xFF98,
    "0xFF99 (sprite Y)":       0xFF99,
    "0xD404 (warp dest X)":    0xD404,
    "0xD405 (warp dest Y)":    0xD405,
    # --- Room/Map candidates ---
    "0x00EB (raw $EB)":        0x00EB,
    "0xC0EB (WRAM $EB)":       0xC0EB,
    "0xFFEB (HRAM $EB)":       0xFFEB,
    "0xFFF6 (map)":            0xFFF6,
    "0xFFF7 (dungeon)":        0xFFF7,
    "0xD401 (area type)":      0xD401,
    "0xD402 (sub-map)":        0xD402,
    "0xD403 (room number)":    0xD403,
}

if __name__ == "__main__":
    pb = pyboy.PyBoy(PATH, window='SDL2', sound_emulated=False)
    pb.set_emulation_speed(1)

    # Load init state
    with open("init.state", "rb") as f:
        pb.load_state(f)

    try:
        from pynput import keyboard as kb
    except ImportError:
        print("[ERROR] pip install pynput")
        pb.stop()
        exit(1)

    pressed = set()
    KEYMAP = {
        "up":    (WindowEvent.PRESS_ARROW_UP, WindowEvent.RELEASE_ARROW_UP),
        "down":  (WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN),
        "left":  (WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT),
        "right": (WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT),
        "z":     (WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A),
        "x":     (WindowEvent.PRESS_BUTTON_B, WindowEvent.RELEASE_BUTTON_B),
    }

    def on_press(key):
        try:
            k = key.char.lower() if hasattr(key, "char") and key.char else key.name.lower()
        except AttributeError:
            return
        if k == "q":
            return False
        if k not in pressed:
            pressed.add(k)
            pair = KEYMAP.get(k)
            if pair:
                pb.send_input(pair[0])

    def on_release(key):
        try:
            k = key.char.lower() if hasattr(key, "char") and key.char else key.name.lower()
        except AttributeError:
            return
        pressed.discard(k)
        pair = KEYMAP.get(k)
        if pair:
            pb.send_input(pair[1])

    listener = kb.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    print("=" * 90)
    print("RAM ADDRESS DIAGNOSTIC — Move Link around and watch which values change!")
    print("Walk left/right to find X addr, up/down to find Y addr, enter a building for room.")
    print("Press Q to quit.")
    print("=" * 90)

    # Snapshot initial values so we can highlight changes
    prev_values = {name: pb.memory[addr] for name, addr in CANDIDATES.items()}

    frame = 0
    while listener.is_alive():
        pb.tick(render=True)
        frame += 1

        # Print every 15 frames (~4x per second)
        if frame % 15 == 0:
            lines = []
            for name, addr in CANDIDATES.items():
                val = pb.memory[addr]
                changed = "  <<<" if val != prev_values[name] else ""
                lines.append(f"  {name:30s} = {val:>3d}  (0x{val:02X}){changed}")
                prev_values[name] = val

            # Clear screen and reprint
            print("\033[2J\033[H")  # ANSI clear
            print("RAM ADDRESS DIAGNOSTIC  (<<< = changed since last read)")
            print("-" * 70)
            for line in lines:
                print(line)
            print("-" * 70)
            print("Move Link around. Q to quit.")

        time.sleep(1 / 60)

    pb.stop()
    print("\n[DONE] Diagnostic finished.")
