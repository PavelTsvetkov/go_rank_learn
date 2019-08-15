import json

import sgf

from utils import next_player, coord, validate_sgf, travese_sgf, strict_valid_rank


class Goban:

    def __init__(self, size, handicap=None) -> None:
        super().__init__()
        self.size = size
        self.history = []
        self.white_prisoners = 0
        self.black_prisoners = 0
        self.history.append(self.initial_state(handicap))

    def initial_state(self, handicap):
        state = {}

        board = [["o" for x in range(self.size)] for y in range(self.size)]

        if handicap:
            for h in handicap:
                x, y = coord(h)
                board[x][y] = 'b'
            state["to_move"] = 'w'
        else:
            state["to_move"] = 'b'

        state["board"] = board
        state["WP"] = 0
        state["BP"] = 0
        return state

    def move(self, who, point):
        state = self.last_state()
        if who != state["to_move"]:
            raise BaseException("Wrong move order")
        state["move"] = point
        # new_state = copy.deepcopy(state)
        new_state = json.loads(json.dumps(state))
        self.do_move(new_state, who, point)
        self.history.append(new_state)

    def last_state(self):
        return self.history[-1]

    def do_move(self, new_state, who, point):
        brd = new_state["board"]
        x, y = coord(point)
        brd[x][y] = who
        new_state["to_move"] = next_player(who)
        del new_state["move"]
        prisoners = self.eliminate_dead_groups(brd, point, next_player(who))
        if who == "w":
            self.white_prisoners = self.white_prisoners + prisoners
        else:
            self.black_prisoners = self.black_prisoners + prisoners
        new_state["WP"] = self.white_prisoners
        new_state["BP"] = self.black_prisoners

    def eliminate_dead_groups(self, brd, point, who):
        px, py = coord(point)
        visited = set()
        prisoners = 0
        for x, y in self.adjanced(px, py):
            cell = brd[x][y]
            if cell == who and not (x, y) in visited:
                grp = self.dead_group_at(brd, x, y, who, visited)
                if grp:
                    prisoners += len(grp)
                for gx, gy in grp:
                    brd[gx][gy] = "o"
        return prisoners

    def adjanced(self, x, y):
        if x > 0:
            yield x - 1, y
        if y > 0:
            yield x, y - 1
        if x < self.size - 1:
            yield x + 1, y
        if y < self.size - 1:
            yield x, y + 1

    def walk(self, brd, x, y, who, visited):
        cell = brd[x][y]
        if (cell == who or cell == "o") and (x, y) not in visited:
            yield cell, x, y
            if cell != "o":
                for ax, ay in self.adjanced(x, y):
                    yield from self.walk(brd, ax, ay, who, visited)

    def dead_group_at(self, brd, x, y, who, visited):
        grp = set()

        for cell, wx, wy in self.walk(brd, x, y, who, visited):
            if cell == "o":
                return set()
            tp = (wx, wy)
            if cell != "o":
                grp.add(tp)
            visited.add(tp)

        return grp

    def print(self):
        state = self.last_state()
        brd = state["board"]

        for l in brd:
            print("".join(l))

        print("WP:", state["WP"])
        print("BP:", state["BP"])

        print()


def sgf_to_goban(sgf_root):
    validate_sgf(sgf_root)
    props = sgf_root.properties

    handicap = None

    if "AB" in props:
        handicap = props["AB"]

    result = Goban(19, handicap=handicap)

    travese_sgf(sgf_root, result)

    meta_inf = {
        "WR": strict_valid_rank(props["WR"][0]),
        "BR": strict_valid_rank(props["WR"][0]),
        "KM": 0 if "KM" not in props else props["KM"][0]
    }

    return result, meta_inf


def load_goban(file):
    with open(file) as f:
        collection = sgf.parse(f.read())
        root = collection.children[0].root
        return sgf_to_goban(root)