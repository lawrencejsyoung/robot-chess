"""robot_chess: a physical AI chess opponent.

Three cooperating subsystems:

* ``engine``  - the *reasoning* system (Stockfish over UCI, or a built-in
  fallback searcher) that decides which move to play.
* ``vision``  - a camera pipeline that turns pixels into board state and
  infers the human's move.
* ``robot``   - the *embodied action* system: a planner that decomposes a
  chess move into pick/place primitives and a controller (GR00T policy or a
  simulator) that executes them.

``game.GameOrchestrator`` coordinates them and ``metrics`` records the
results so runs can be compared.
"""

__version__ = "0.1.0"
