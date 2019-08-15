from unittest import TestCase

from goban import Goban


class TestGoban(TestCase):
    def test_move(self):
        g = Goban(3, handicap=["aa", "ab", "ac", "ba", "bc", "cc", "cb", "bc", "ca"])

        g.move("w", "bb")

        print(g.history)
