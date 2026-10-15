"""Two-player Connect Four match coordinator for reinforcement learning.

Game advances one stone at a time. A *learning* transition ends when the
same player gets their next turn, or immediately when the game terminates.

Agents must provide:
    get_action(state: np.ndarray, legal_mask: np.ndarray) -> int

Agents may additionally provide:
    observe(transition: Transition) -> None

For a frozen opponent, implement only get_action(). The Match class never
calls learn()/backward(): each agent decides how to store or learn from data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from connect4 import Connect4


class Agent(Protocol):
    def get_action(self, state: np.ndarray, legal_mask: np.ndarray) -> int:
        """Choose a legal column; board uses +1 for own stones."""
        ...


@dataclass(frozen=True)
class Transition:
    """One *player-turn* transition for DQN, not one board move.

    `state` and `next_state` are both from `player`'s perspective.
    `done` is True if the match ended before this player's next turn.
    For terminal experiences, `next_legal_mask` is all False.
    """

    player: int
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    next_legal_mask: np.ndarray


@dataclass(frozen=True)
class MatchResult:
    winner: int            # 1: first player; -1: second player; 0: draw
    actions: tuple[int, ...]
    transitions: tuple[Transition, ...]

    @property
    def move_count(self) -> int:
        return len(self.actions)


@dataclass
class _Pending:
    state: np.ndarray
    action: int


class Match:
    """Play a complete match, routing finished experiences to each agent.

    - Nonterminal experience: (board before own move, own move, 0,
      board after opponent's reply, False, legal mask for next own move).
    - Win: terminal experience for winner (+1) and loser (-1).
    - Draw: terminal experience for both (0).
    - A frozen agent can omit observe().

    The clock for gamma is one *player turn*, i.e. usually two board moves.
    """

    def __init__(self, game: Connect4, first_agent: Agent, second_agent: Agent):
        self.game = game
        self.agents = {1: first_agent, -1: second_agent}

    def play(self, render: bool = False) -> MatchResult:
        """Reset the board, play to a terminal result, and deliver transitions.

        The match's agents remain as given; only the game is reset. This is
        important for self-play agents that maintain a replay buffer/model.
        """
        self.game.reset()

        pending: dict[int, _Pending] = {}
        actions: list[int] = []
        transitions: list[Transition] = []

        if render:
            self.game.render()

        def complete(player: int, reward: float, done: bool) -> None:
            old = pending.pop(player)
            transition = Transition(
                player=player,
                state=old.state,
                action=old.action,
                reward=float(reward),
                next_state=self.game.observe(player),
                done=done,
                next_legal_mask=self.game.legal_action_mask().copy(),
            )
            transitions.append(transition)

            observer = getattr(self.agents[player], "observe", None)
            if observer is not None:
                observer(transition)

        while not self.game.done:
            player = self.game.current_player
            agent = self.agents[player]

            # Separate the agent's input from the saved state. In particular,
            # an agent must not be able to mutate the pending state by writing
            # into the array passed to get_action().
            state = self.game.observe(player)
            legal_mask = self.game.legal_action_mask().copy()
            action = agent.get_action(state.copy(), legal_mask.copy())

            # Game checks range, dtype, and whether the column is full.
            _, done, winner = self.game.step(action)
            actions.append(int(action))
            pending[player] = _Pending(state=state, action=int(action))

            if render:
                self.game.render()

            opponent = -player
            if done:
                reward = 0.0 if winner == 0 else 1.0
                complete(player, reward, done=True)

                # The loser (or the other player on a draw) still has a
                # pending action whose outcome has now been decided.
                if opponent in pending:
                    complete(opponent, -reward, done=True)
            else:
                # Opponent just received our reply, so its pending experience
                # now ends at the opponent's next turn.
                if opponent in pending:
                    complete(opponent, 0.0, done=False)

        assert not pending, "no unfinished experiences should remain"
        assert self.game.winner is not None

        return MatchResult(
            winner=int(self.game.winner),
            actions=tuple(actions),
            transitions=tuple(transitions),
        )
