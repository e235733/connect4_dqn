"""Connect Four game rules (no agents, rewards, or rendering dependencies).

Board convention:
    S[r * width + c], where row 0 is the BOTTOM row.
    1: first player (O), -1: second player (X), 0: empty.
"""

from __future__ import annotations

import numpy as np


class Connect4:
    """Turn-based Connect Four game, separate from the RL training loop.

    Public API:
        reset()                   -> copy of the empty absolute board
        step(column)              -> (board_copy, done, winner)
        legal_actions()           -> list of available column numbers
        legal_action_mask()       -> bool array of shape (width,)
        observe(player=None)      -> absolute board, or player-relative board
        render()                  -> print and return a readable board

    Winner: None during play; +1/-1 when that player wins; 0 on a draw.
    `done` becomes True on a win or draw. Game rewards are NOT calculated here.
    """

    def __init__(self, height: int = 6, width: int = 7):
        if (not isinstance(height, (int, np.integer)) or isinstance(height, (bool, np.bool_))
                or not isinstance(width, (int, np.integer)) or isinstance(width, (bool, np.bool_))
                or height < 1 or width < 1):
            raise ValueError("height and width must be positive integers")

        self.height = int(height)
        self.width = int(width)

        # Keep the original one-dimensional, bottom-row-first representation.
        self.S = np.zeros(self.height * self.width, dtype=np.int8)
        self.roof = np.zeros(self.width, dtype=np.int32)

        self.current_player = 1
        self.move_count = 0
        self.done = False
        self.winner: int | None = None

    def idx_S(self, r: int, c: int) -> int:
        """Convert a bottom-based (row, column) pair into a flat index."""
        if not (0 <= r < self.height and 0 <= c < self.width):
            raise IndexError(f"cell out of bounds: (r={r}, c={c})")
        return r * self.width + c

    def reset(self) -> np.ndarray:
        self.S.fill(0)
        self.roof.fill(0)
        self.current_player = 1
        self.move_count = 0
        self.done = False
        self.winner = None
        return self.S.copy()

    def observe(self, player: int | None = None) -> np.ndarray:
        """Return an independent board snapshot.

        player=None: absolute board (+1 first, -1 second).
        player=+1/-1: board from that player's perspective (+1 own stones).
        """
        if player is None:
            return self.S.copy()
        if player not in (1, -1):
            raise ValueError("player must be +1, -1, or None")
        return (self.S * player).copy()

    def legal_actions(self) -> list[int]:
        if self.done:
            return []
        return np.flatnonzero(self.roof < self.height).tolist()

    def legal_action_mask(self) -> np.ndarray:
        """True means the column is selectable; always False after game over."""
        if self.done:
            return np.zeros(self.width, dtype=bool)
        return self.roof < self.height

    def judge(self, r: int, c: int, is_first: bool) -> bool:
        """Does the given player's stone at (r, c) complete four in a row?"""
        p = 1 if is_first else -1
        if self.S[self.idx_S(r, c)] != p:
            return False

        # Vertical, horizontal, rising diagonal, falling diagonal.
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for dr, dc in directions:
            chain = 1  # The stone just placed at (r, c).

            for sign in (1, -1):
                for offset in (1, 2, 3):
                    nr = r + dr * sign * offset
                    nc = c + dc * sign * offset

                    if not (0 <= nr < self.height and 0 <= nc < self.width):
                        break
                    if self.S[self.idx_S(nr, nc)] != p:
                        break
                    chain += 1

            if chain >= 4:
                return True

        return False

    def step(self, column: int) -> tuple[np.ndarray, bool, int | None]:
        """Place one stone for current_player and advance the game.

        Returns (absolute_board_copy, done, winner).
        Raises on out-of-range/full-column moves or attempts after game over.
        """
        if self.done:
            raise RuntimeError("game is already finished; call reset()")
        if isinstance(column, (bool, np.bool_)) or not isinstance(column, (int, np.integer)):
            raise TypeError("column must be an integer")
        if not 0 <= column < self.width:
            raise ValueError(f"column must be in range [0, {self.width - 1}]")
        if self.roof[column] >= self.height:
            raise ValueError(f"column {column} is full")

        r = int(self.roof[column])
        p = self.current_player
        self.S[self.idx_S(r, column)] = p
        self.roof[column] += 1
        self.move_count += 1

        if self.judge(r, column, is_first=(p == 1)):
            self.done = True
            self.winner = p
        elif self.move_count == self.height * self.width:
            self.done = True
            self.winner = 0
        else:
            self.current_player = -p

        return self.S.copy(), self.done, self.winner

    def action(self, column: int, is_first: bool) -> tuple[np.ndarray, bool, int | None]:
        """Backward-compatible wrapper for the original `action` method.

        Validates that the passed player matches the actual turn. New code should
        call `step(column)` instead, avoiding duplicate turn management.
        """
        expected = (self.current_player == 1)
        if bool(is_first) != expected:
            raise ValueError("is_first does not match the current turn")
        return self.step(column)

    def render(self) -> str:
        """Show top row first, preserving the original O/X convention."""
        symbols = {0: " ", 1: "o", -1: "x"}
        lines = []
        for r in range(self.height - 1, -1, -1):
            row = "".join(symbols[int(self.S[self.idx_S(r, c)])]
                          for c in range(self.width))
            lines.append(f"|{row}|")
        lines.append("+" + "-" * self.width + "+")
        lines.append(" " + "".join(str(c % 10) for c in range(self.width)))
        picture = "\n".join(lines)
        print(picture)
        return picture


# Compatibility with the class name from the original snippet.
connect4 = Connect4
