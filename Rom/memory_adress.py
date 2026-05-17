# ─── Player Position ──────────────────────────────────────────────────────────
# 0xEB  = Overworld room ID:  value = room_x + 0x10 * room_y (16x8 grid)
#         This is the canonical address to track WHICH overworld screen Link is on.
OVERWORLD_ROOM = 0xEB         # Current room on the overworld grid (0x00 – 0x7F)

# 0xFF98 / 0xFF99 = Link's pixel position within the current room/screen
#   These are the ONLY addresses that update in real-time as Link moves.
#   X: 0–160 pixels (full screen width)
#   Y: 0–144 pixels (full screen height)
PLAYER_X_INROOM = 0xFF98      # Link's in-room X pixel position
PLAYER_Y_INROOM = 0xFF99      # Link's in-room Y pixel position
# Backward-compat aliases (old code references these names)
PLAYER_X = PLAYER_X_INROOM
PLAYER_Y = PLAYER_Y_INROOM
GREAT_PLAYER_X = PLAYER_X_INROOM  # Legacy alias
GREAT_PLAYER_Y = PLAYER_Y_INROOM  # Legacy alias

# 0xFFF6 = Current map/room ID (changes when Link walks between screens)
MAP_ROOM_ID = 0xFFF6

# ─── Map / World Info ─────────────────────────────────────────────────────────
# 0xD401 = Map type:  0=Overworld, 1=Dungeon/indoor, 2=Side-scroll
CURRENT_WORLD = 0xD401        # Area type (overworld vs indoor vs side-scroll)
# 0xD402 = Sub-map (dungeon number, 0x00-0x1F; 0xFF = Color Dungeon)
CURRENT_DUNGEON = 0xD402
# 0xD403 = Room number within the current dungeon/area
ROOM_NUMBER = 0xD403



CURRENT_LOADED_MAP = [0xD700, 0xD79B]
WORLD_STATUT = [0xD800, 0xD8FF] # * min - max In reality idk, It seems that when you discoverd a new location it's setting it to 128
HELD_ITEM_1 = 0xDB00 # Refered to the item's index list
HELD_ITEM_2 = 0xDB01 # Refered to the item's index list
INVENTORY = [0xDB02, 0xDB0B]
NUMBER_BOMBS = 0xDB4D # Seems to work
SHIELD_LEVEL = 0xDB44 # 0 if we don't take it, 1 else and 2 if hold.
SWORD_LEVEL = 0xDB4E # Same as the shield
NUMBER_ARROWS = 0xDB45 # Seems to work
MAX_BOMB = 0xDB77 # Seems to work
MAX_ARROWS = 0xDB78 # Seems to work
CURRENT_HEALTH = 0xDB5A # Works in a scale of 24, so each heart represent 6 hp
KILLED_MONSTERS = [0xDBB5, 0xF415] # Finally found!!!!!
NUMBERR_DEATH = 0xDB56 # ! TBD


"""
Items list:

------------------------------
|   item           |    id    |
|------------------|----------|
|  Sword           |    01    |
|  Bombs           |    02    |
|  Power Bracelet  |    03    |
|  Shield          |    04    |
|  Bow             |    05    |
|  HookShot        |    06    |
|  Fire Rod        |    07    |
|  Pegasus boots   |    08    |
|  Ocarina         |    09    |
|  Feather         |    0A    |
|  Shovel          |    0B    |
|  Magic Powder    |    0C    |
|  Boomrang        |    0D    |
------------------------------

"""