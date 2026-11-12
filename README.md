# Robot Chess

A physical AI chess opponent prototype coordinating **perception → chess reasoning → embodied action**. Play a complete game against a local engine in a responsive web dashboard and inspect the physical task plan for every robot move.

## Run

Python 3.12 or newer:

```sh
cd /workspace/robot-chess
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py
```

Open port 5000 in your development environment. Run tests with `.venv/bin/python -m pytest -q`.

The default engine is a local depth-three alpha-beta material search. Set `STOCKFISH_PATH` to an installed Stockfish executable to use its UCI engine instead. No API keys or external services are required for simulation. White is human, black is AI; promotion in the dashboard automatically selects a queen (the API accepts other promotion choices).

## What works

- Legal move validation, check, checkmate, draws and move history using python-chess.
- An engine response for each human move, with a retry button if a turn fails.
- Physical task planning for transfers, captured pieces, en passant, castling and promotion.
- FEN observations checked against the current occupancy and all legal next positions without mutating game state.
- Responsive dashboard with legal destination highlights and a readable action plan.

## Camera and GR00T 2 integration

**This is a simulation, not an integrated robot controller.** No camera recognition or GR00T inference is bundled. An embodied policy is hardware-specific; do not send these symbolic tasks directly to motors. The UI identifies both integrations as disconnected.

A camera integration should calibrate the board plane and orientation, classify pieces on each square, and send a full FEN to `POST /api/observe` as `{"fen":"..."}`. The endpoint returns `synchronized` and `matches` (legal UCI moves explaining the occupancy). A vision controller must require one stable, unique match and confidence thresholds before submitting it to `/api/move`. FEN side-to-move fields are not inferred by image recognition; the coordinator owns game state. The current observation endpoint compares occupancy only.

`plan_move` in `app.py` is the GR00T 2 adapter boundary. It emits ordered symbolic tasks:

```json
[
  {"action":"transfer","from":"e7","to":"e5"},
  {"action":"verify","expected_fen":"..."}
]
```

An external deployment should map board squares and capture trays to calibrated poses, attach camera/proprioception observations, and translate tasks into the input schema of its installed NVIDIA Isaac GR00T policy runtime. It needs the appropriate model checkpoint, embodiment configuration, GPU runtime, robot drivers and a calibrated grasp/release policy. This project deliberately does not invent an inference API or assume a compatible checkpoint.

Before enabling hardware, implement an acknowledged execution loop: observe → validate human move → plan AI move → execute with workspace bounds and emergency stop → verify observed board → commit game state. **The simulator commits immediately**; use a separate pending/acknowledged state machine for hardware, with timeouts and recovery for failed grasps. Promotion currently requires physical piece replacement. Test captures, castling and recovery on the actual rig before autonomous play.

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /api/state` | Position, legal moves, engine, history, latest plan |
| `POST /api/move` | Submit human UCI move: `{"move":"e2e4"}` |
| `POST /api/robot` | Plan and apply a simulated black move |
| `POST /api/observe` | Validate external FEN observation |
| `POST /api/reset` | Start a new game |

This development server holds one shared game in memory. Restarting clears it. It has no authentication or persistence and is intended for a local lab session, not a public multi-user service. Fonts have local fallbacks and no external access is needed for chess behavior.

## Cloud development

Use the existing checkout: each cloud task is already isolated; do not create a Git worktree unless requested. Dependencies persist in `.venv`; restart the server in each new task. Readiness: `GET /api/state` should return the starting position and 20 legal moves. Run the test suite before changes and after modifications to coordination logic.
