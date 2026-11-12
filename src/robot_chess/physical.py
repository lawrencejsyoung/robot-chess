"""A model of where every physical piece is, on and off the board.

Locations are either square names ("e4") or named off-board zones
("graveyard_white", "reserve", ...). The simulator mutates this model and the
simulated camera renders it, so the vision stack sees the consequences of the
robot's (possibly imperfect) actions.
"""

from __future__ import annotations

from collections import defaultdict

import chess

ON_BOARD = set(chess.SQUARE_NAMES)


def is_square(location: str) -> bool:
    return location in ON_BOARD


class PhysicalBoard:
    def __init__(self) -> None:
        self.squares: dict[str, chess.Piece] = {}
        self.zones: dict[str, list[chess.Piece]] = defaultdict(list)

    @classmethod
    def from_board(cls, board: chess.Board, reserve: str = "reserve") -> "PhysicalBoard":
        pb = cls()
        for sq, piece in board.piece_map().items():
            pb.squares[chess.square_name(sq)] = piece
        # A spare queen of each colour for promotions.
        pb.zones[reserve] = [chess.Piece(chess.QUEEN, chess.WHITE), chess.Piece(chess.QUEEN, chess.BLACK)]
        return pb

    def copy(self) -> "PhysicalBoard":
        pb = PhysicalBoard()
        pb.squares = dict(self.squares)
        for k, v in self.zones.items():
            pb.zones[k] = list(v)
        return pb

    def piece_at(self, location: str) -> chess.Piece | None:
        if is_square(location):
            return self.squares.get(location)
        return self.zones[location][-1] if self.zones.get(location) else None

    def take(self, location: str, piece: chess.Piece | None = None) -> chess.Piece | None:
        """Remove and return a piece. In zones, ``piece`` selects which one."""
        if is_square(location):
            return self.squares.pop(location, None)
        zone = self.zones.get(location, [])
        if piece is not None:
            for i in range(len(zone) - 1, -1, -1):
                if zone[i] == piece:
                    return zone.pop(i)
            return None
        return zone.pop() if zone else None

    def put(self, location: str, piece: chess.Piece) -> None:
        if is_square(location):
            if location in self.squares:
                raise ValueError(f"Square {location} already occupied")
            self.squares[location] = piece
        else:
            self.zones[location].append(piece)

    def occupancy(self) -> list[int]:
        """64-entry list indexed by chess.Square: 0 empty, 1 white, 2 black."""
        return occupancy_from_pieces(
            {chess.parse_square(n): p for n, p in self.squares.items()}
        )

    def matches(self, board: chess.Board) -> bool:
        return {chess.parse_square(n): p for n, p in self.squares.items()} == board.piece_map()


EMPTY, WHITE, BLACK = 0, 1, 2


def occupancy_from_pieces(piece_map: dict[int, chess.Piece]) -> list[int]:
    occ = [EMPTY] * 64
    for sq, p in piece_map.items():
        occ[sq] = WHITE if p.color == chess.WHITE else BLACK
    return occ


def occupancy_from_board(board: chess.Board) -> list[int]:
    return occupancy_from_pieces(board.piece_map())
