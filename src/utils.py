import numpy as np

# ── Shared constants ───────────────────────────────────────────────────────────
DATA_DIR      = "data"
N_BITS        = 10    # bits in player input (max value 1023)
HISTORY       = 30    # frames of history for BC sliding-window observation
ACTION_BITS   = 11    # 2^11 = 2048 covers up to 2048 action IDs
MAX_ACTION_ID = 2048  # embedding table size (2^ACTION_BITS)
FLOAT_DIM     = 18   # float features per frame returned by frame_features

# ── Action ID canonicalization ─────────────────────────────────────────────────

_ACTION_ID_MAP = {
    6: 1,                                                               # standing
    14: 13, 15: 13,                                                     # backwards
    10: 9, 11: 9,                                                       # forward
    5: 4,                                                               # crouch
    34: 33, 35: 33, 36: 33, 37: 33, 38: 33, 39: 33, 40: 33, 41: 33,     # airborne
    489: 480,                                                           # parry
    154: 153, 155: 153, 160: 153, 161: 153, 162: 153, 167: 153,         # high block
    182: 181, 174: 181, 175: 181, 176: 181,                             # low block
    250: 721, 350: 721, 342: 721, 330: 721,                             # grabbed
    483: 482, 484: 482, 485: 482,                                       # parry block
}

def _canonical_action_id(action_id: int) -> int:
    return _ACTION_ID_MAP.get(action_id, action_id)

# ── Feature extraction ─────────────────────────────────────────────────────────

def _get_player_and_opponent(frame: dict):
    """Returns (cammy, opponent) regardless of which side Cammy is on."""
    if frame["p1"]["id"] == 9:
        return frame["p1"], frame["p2"]
    return frame["p2"], frame["p1"]

def frame_features(frame: dict) -> tuple[list, list]:
    """Returns (float_features, [player_action_id, opponent_action_id]).
    float_features: 18 floats (positions, distance, stance, y-diff, HP, drive, super, distance buckets).
    Action IDs are kept separate for nn.Embedding lookup in the LSTM model."""
    player, opponent = _get_player_and_opponent(frame)
    mirror = not bool(player["dir"])
    sign   = -1.0 if mirror else 1.0
    stance = opponent["stance"]

    dist_norm = float(np.clip(player["dist"] / 490.0, 0.0, 1.0))
    float_features = [
        np.clip(sign * player["pos"]["x"]   / 765.0, -1.0, 1.0),
        np.clip(       player["pos"]["y"]   / 300.0,  0.0, 1.0),
        np.clip(sign * opponent["pos"]["x"] / 765.0, -1.0, 1.0),
        np.clip(       opponent["pos"]["y"] / 300.0,  0.0, 1.0),
        dist_norm,
        float(stance == 0),
        float(stance == 1),
        float(stance == 2),
        np.clip(abs(player["pos"]["y"] - opponent["pos"]["y"]) / 300.0, 0.0, 1.0),
        np.clip(player["hp"]            / 10000.0, 0.0, 1.0),
        np.clip(opponent["hp"]          / 10000.0, 0.0, 1.0),
        np.clip(player["drive"]         / 60000.0, 0.0, 1.0),
        np.clip(opponent["drive"]       / 60000.0, 0.0, 1.0),
        np.clip(player["super"]         / 30000.0, 0.0, 1.0),
        np.clip(opponent["super"]       / 30000.0, 0.0, 1.0),
        float(dist_norm < 0.25),          # close range  (< ~130px)
        float(0.30 <= dist_norm < 0.50),  # medium range
        float(dist_norm >= 0.50),         # far range    (> ~294px)
    ]
    action_ids = [_canonical_action_id(int(player["action"][0])),
                  _canonical_action_id(int(opponent["action"][0]))]
    return float_features, action_ids

def extract_obs(window: list[dict]) -> np.ndarray:
    """Flat float32 vector of shape (HISTORY * 52,).
    Pads with copies of the first frame when the window is shorter than HISTORY."""
    while len(window) < HISTORY:
        window = [window[0]] + window
    features = []
    for frame in window:
        floats, action_ids = frame_features(frame)
        p_bits   = [float((action_ids[0] >> i) & 1) for i in range(ACTION_BITS)]
        opp_bits = [float((action_ids[1] >> i) & 1) for i in range(ACTION_BITS)]
        features += p_bits + floats[:2] + opp_bits + floats[2:4] + floats[4:]
    return np.array(features, dtype=np.float32)

def extract_obs_embed(window: list[dict]) -> np.ndarray:
    """Flat float32 vector of shape ((2 + 9) * HISTORY,) for the embedding BC model.
    First 2*HISTORY values are raw action IDs stored as float32 (cast to long in extractor).
    Last 9*HISTORY values are normalized float features."""
    while len(window) < HISTORY:
        window = [window[0]] + window
    ids_list, floats_list = [], []
    for frame in window:
        floats, action_ids = frame_features(frame)
        ids_list.extend(action_ids)
        floats_list.extend(floats)
    return np.array(ids_list + floats_list, dtype=np.float32)

def extract_action(frame: dict) -> np.ndarray:
    """Cammy's input as a float32 bit array of shape (N_BITS,).
    LEFT and RIGHT bits are swapped when Cammy is on the right side."""
    player, _ = _get_player_and_opponent(frame)
    val = _clean_input(player["input"])
    bits = [(val >> i) & 1 for i in range(N_BITS)]
    if not bool(player["dir"]):  # Cammy on right side — mirror directional inputs
        bits[2], bits[3] = bits[3], bits[2]  # swap LEFT and RIGHT
    return np.array(bits, dtype=np.float32)

def _clean_input(val: int) -> int:
    """Zero out attack bits (4-9) if more than 3 are pressed simultaneously."""
    _ATTACK_MASK = 0b1111110000  # bits 4-9
    if bin(val & _ATTACK_MASK).count('1') > 3:
        val &= ~_ATTACK_MASK
    return val

def _mirror_input(val: int) -> int:
    """Swap LEFT (bit 2) and RIGHT (bit 3) in a raw input integer."""
    left  = (val >> 2) & 1
    right = (val >> 3) & 1
    return (val & ~0b1100) | (right << 2) | (left << 3)

def build_action_id_vocab(frames: list) -> tuple[dict[int, int], int]:
    """Scans all frames and builds a contiguous vocab of canonical action IDs.
    Returns (vocab: canonical_id→index, n_ids). Index 0 is reserved for unknown IDs."""
    seen = set()
    for f in frames:
        for p in [f["p1"], f["p2"]]:
            seen.add(_canonical_action_id(int(p["action"][0])))
    vocab = {aid: i + 1 for i, aid in enumerate(sorted(seen))}
    return vocab, len(seen) + 1  # +1 for the reserved unknown slot

def build_input_vocab(frames: list) -> tuple[dict, list]:
    """Scans all frames and builds a vocabulary of observed mirrored input values.
    Returns (vocab: raw_int→class_idx, input_ints: list indexed by class).
    Class 0 is always no-input (value 0) so it's safe to use as padding."""
    seen = set()
    for f in frames:
        player, _ = _get_player_and_opponent(f)
        val = _clean_input(player["input"])
        if not bool(player["dir"]):
            val = _mirror_input(val)
        seen.add(val)
    input_ints = [0] + sorted(seen - {0})  # class 0 = no input
    vocab = {v: i for i, v in enumerate(input_ints)}
    return vocab, input_ints

def extract_action_idx(frame: dict, vocab: dict) -> int:
    """Returns the vocab class index for Cammy's input in this frame."""
    player, _ = _get_player_and_opponent(frame)
    val = _clean_input(player["input"])
    if not bool(player["dir"]):
        val = _mirror_input(val)
    return vocab.get(val, 0)

# ── Episode segmentation ───────────────────────────────────────────────────────

def _is_default_state(frame: dict) -> bool:
    """True for pre-fight idle frames: both players at full HP and default positions."""
    p1, p2 = frame["p1"], frame["p2"]
    full_hp = p1["hp"] >= 10000 and p2["hp"] >= 10000
    default_pos = abs(p1["pos"]["x"]) == 150 and abs(p2["pos"]["x"]) == 150
    #default_action = p1["action"][0] == 1 and p2["action"][0] == 1
    return full_hp and default_pos # and default_action

def _is_end_state(frame: dict) -> bool:
    return frame["p1"]["hp"] <= 0 or frame["p2"]["hp"] <= 0

def split_episodes(frames: list) -> list[list[dict]]:
    """Segments a flat list of frames into per-round episode lists."""
    prev = None
    clean_frames = []
    for f in frames:
        if _is_end_state(f):
            prev = f
            continue
        if _is_default_state(f) and prev is not None and _is_default_state(prev):
            prev = f
            continue
        clean_frames.append(f)
        prev = f

    episodes = []
    new_round = True
    ep = []
    for f in clean_frames:
        if new_round and not _is_default_state(f):
            new_round = False
        if not new_round and _is_default_state(f):
            if len(ep) > 300:
                episodes.append(ep)
            new_round = True
            ep = []
        ep.append(f)
    if len(ep) > 300:
        episodes.append(ep)

    return episodes
