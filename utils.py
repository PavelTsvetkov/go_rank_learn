import re

import numpy as np

from constants import META_KEY, HISTORY_KEY

rank_re = re.compile("\\dp")


def normalize_rank(rnk):
    if "," in rnk:
        rnk = rnk.split(",")[0]
    return rnk


def is_valid_rank(rank):
    return rank_re.fullmatch(rank)


def strict_valid_rank(rank):
    rnk = normalize_rank(rank)
    if is_valid_rank(rnk):
        return rnk
    else:
        raise BaseException("Invalid rank:" + rnk)


REQ_PROPS = {"WR", "BR"}


def validate_sgf(sgf_root):
    props = sgf_root.properties
    dif_keys = REQ_PROPS.difference(props.keys())
    if len(dif_keys):
        raise BaseException("Missing keys:" + str(dif_keys))

    if "RU" in props and props["RU"][0] != "Japanese":
        raise BaseException("Unsupported rules:" + props["RU"])
    if "SZ" in props and props["SZ"][0] != 19:
        raise BaseException("Unsupported size:" + props["SZ"])

    if "AW" in props:
        raise BaseException("Game has AW tag")

    if "AB" in props:
        if not "HA" in props:
            raise BaseException("AB without HA")
        ha = int(props["HA"][0])
        ab = props["AB"]
        if ha != len(ab):
            raise BaseException("Size of AB not same as HA")


def travese_sgf(root, goban):
    next = root.next
    keys = {"W", "B"}
    while next:
        props = next.properties
        common_key = keys.intersection(props.keys())
        if len(common_key) != 1:
            raise BaseException("Expected either B or W, but found:" + str(common_key))
        who = common_key.pop()
        move = props[who][0]
        goban.move(who.lower(), move)
        next = next.next


# goban, meta = load_goban("C:\\tmp\\kgs_learn_workdir\\games\\ancient\\Honinbo_Jowa\\215.sgf")
#
# goban.print()
#
# print(meta)
coord_map = {chr(k): v for k, v in zip(range(ord('a'), ord('z') + 1), range(0, 26))}


def next_player(who):
    if who == "b":
        return "w"
    if who == "w":
        return "b"
    raise BaseException("Unknown player: " + who)


def coord(h):
    return coord_map[h[0]], coord_map[h[1]]


DAN_LIST = [str(x) + 'p' for x in range(1, 10)]
DAN_MAP = {dan: idx for idx, dan in enumerate(DAN_LIST)}

CLASS_WEIGHTS = {0: 30.30873622, 1: 27.55127217, 2: 24.2100271, 3: 24.93649686, 4: 15.32332762, 5: 12.26707861,
                 6: 8.285184326, 7: 8.596102959,
                 8: 2.148250571}


def get_smp_new(board, mv, to_move):
    brd = np.zeros((19, 19), dtype=np.float32)
    move = np.zeros((19, 19), dtype=np.float32)
    for x, line in enumerate(board):
        brd[x] = np.array([encode_board(z, to_move) for z in line], dtype=np.float32)
    x, y = coord(mv)
    move[x][y] = 1.0
    return brd, move


def get_smp_alpha(board, cell_color):
    brd = np.zeros((19, 19), dtype=np.float32)
    for x, line in enumerate(board):
        brd[x] = np.array([1.0 if z == cell_color else 0.0 for z in line], dtype=np.float32)
    return brd


def encode_board(brd_point, to_move):
    if brd_point == "o":
        return 0.0
    return 1.0 if brd_point == to_move else -1.0


def game_to_numpy(game, black_mode=False):
    meta = game[META_KEY]
    history = game[HISTORY_KEY]

    board_states = np.zeros(shape=(len(history) - 1, 19, 19), dtype=np.float32)
    moves = np.zeros(shape=(len(history) - 1, 19, 19), dtype=np.float32)
    dans = np.zeros(shape=(len(history) - 1, len(DAN_LIST)), dtype=np.float32)

    for idx, h in enumerate(history):
        if not ("move" in h):
            break
        dan_map = {
            "w": meta["WR"],
            "b": meta["BR"],
        }
        to_move = h["to_move"]
        dan = dan_map[to_move]
        if not dan in DAN_MAP:
            raise BaseException("Unknown dan ", dan)
        dan_idx = DAN_MAP[dan]
        board, move = get_smp_new(h["board"], h["move"], "b" if black_mode else to_move)
        labels = np.zeros((len(DAN_LIST)), dtype=np.float32)
        labels[dan_idx] = 1.0
        board_states[idx] = board
        moves[idx] = move
        dans[idx] = labels
    return {"board_states": board_states, "moves": moves, "dans": dans}


def game_to_numpy_alpha(game):
    meta = game[META_KEY]
    history = game[HISTORY_KEY]

    if len(history) < 3:
        raise BaseException("Game is too short")

    empty_cells = np.zeros(shape=(len(history) - 1, 19, 19), dtype=np.float32)
    black_cells = np.zeros(shape=(len(history) - 1, 19, 19), dtype=np.float32)
    white_cells = np.zeros(shape=(len(history) - 1, 19, 19), dtype=np.float32)
    moves = np.zeros(shape=(len(history) - 1, 19, 19), dtype=np.float32)
    white_dans = np.zeros(shape=(len(DAN_LIST)), dtype=np.float32)
    black_dans = np.zeros(shape=(len(DAN_LIST)), dtype=np.float32)
    black_to_move = np.zeros(shape=(len(history) - 1), dtype=np.float32)

    white_dans[DAN_MAP[meta["WR"]]] = 1.0
    black_dans[DAN_MAP[meta["BR"]]] = 1.0

    for idx, h in enumerate(history):
        if not ("move" in h):
            break
        if h["to_move"] == "b":
            black_to_move[idx] = 1.0
        board = h["board"]
        empty_cells[idx] = get_smp_alpha(board, "o")
        black_cells[idx] = get_smp_alpha(board, "b")
        white_cells[idx] = get_smp_alpha(board, "w")
        mx, my = coord(h["move"])
        moves[idx][mx][my] = 1.0

    return {
        "empty_cells": empty_cells,
        "black_cells": black_cells,
        "white_cells": white_cells,
        "moves": moves,
        "white_dans": white_dans,
        "black_dans": black_dans,
        "black_to_move": black_to_move
    }
