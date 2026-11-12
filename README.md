# robot-chess: a physical AI chess opponent

A robot that plays over-the-board chess against you. It combines three AI systems:

| Role | Component | What it does |
|---|---|---|
| **Reasoning** | `engine.py` | Picks the robot's move with Stockfish over UCI, or a built-in alpha-beta searcher if Stockfish isn't installed |
| **Perception** | `vision/` | Finds the board in the camera image, works out which squares are empty, white or black, and infers your move |
| **Action** | `robot/` | Breaks a chess move into language-conditioned pick-and-place steps and runs them with an NVIDIA **GR00T** policy |

`game.py` coordinates the three. After every primitive, **vision checks what the robot did**. If the step failed, the robot plans a correction and retries. If retries run out, it asks you for help. Every move is logged, so you can measure and compare results across runs.

```
             ┌─────────────┐   move    ┌──────────────┐  PickPlace[]  ┌──────────────────┐
  camera ───▶│   vision    │──────────▶│    engine    │──────────────▶│  planner + GR00T │──▶ arm
             │ occupancy + │  (human)  │ (Stockfish)  │               │  "pick the black │
             │ legal-move  │           └──────────────┘               │  knight on g8 …" │
             │  matching   │◀──────────── verify after each primitive ─┴──────────────────┘
             └─────────────┘                 │ mismatch → correction / retry / ask human
                                             ▼
                                     metrics (JSONL + summary)
```

## Quickstart (simulation, no hardware)

The whole stack runs in simulation. A renderer draws the board as a tilted camera would see it, and the same vision code that runs on hardware reads it. A simulated robot moves pieces and fails at a rate you choose: it can miss a grasp or drop a piece on the wrong square. The orchestrator only finds those failures through vision.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"          # add ".[groot]" for the real GR00T client
pytest -q

# 10 games against a random-move "human", 10% robot failure rate per primitive
robot-chess benchmark --games 10 --failure-rate 0.1

# Play against it yourself in the simulator (type moves as SAN or UCI)
robot-chess play --sim
```

## Measurable results

Each game writes `runs/<name>/game-NNN.jsonl` (one record per ply) and a `.summary.json`. A benchmark also writes `benchmark.json`:

| Metric | Meaning |
|---|---|
| `robot_first_try_rate` | Share of robot moves completed with no corrections and no human help |
| `robot_autonomous_rate` | Share of robot moves vision-verified without human help (corrections allowed) |
| `corrections_per_robot_move` | Self-recoveries the verify loop triggered per move |
| `human_assists` | Times the robot gave up and asked you to fix the board |
| `mean_execution_time_s` | Wall time to physically carry out a move |
| `detection_accuracy` | Share of human moves vision identified correctly (ground truth only exists in sim) |
| `mean_detection_latency_s` | Time from your move to the robot recognising it |
| `mean_engine_time_s`, `score_cp` | Reasoning cost and evaluation |

Example: `robot-chess benchmark --games 2 --max-plies 40 --failure-rate 0.15` gives a first-try rate of about 0.85, an autonomous rate of 1.0 and detection accuracy of 1.0. Use these numbers as a sim baseline, then compare them with hardware runs and with policy checkpoints.

## How it works

**Vision.** `robot-chess calibrate` finds the 7×7 inner corners of an *empty* board, computes a homography to a top-down view, and saves an empty-board reference image. Each square's centre crop is compared with that reference: pixels that differ belong to a piece, and their brightness separates white from black. Occupancy alone can't tell a knight from a bishop, and it doesn't need to. The detector looks for the **legal move** whose resulting occupancy matches what the camera sees, so castling, en passant and captures are handled without extra code. It only trusts a reading after several identical frames in a row, so a hand passing over the board doesn't count as a move. If you need more robustness, `OccupancyClassifier` can be swapped for a small CNN with the same interface.

**Planning.** `MovePlanner` turns a move into ordered `PickPlace(piece, src, dst)` steps. A capture removes the victim to its graveyard first. Castling moves the king, then the rook. En passant removes the pawn beside the destination. Promotion sends the pawn to the graveyard and brings a queen from the reserve.

**Action (GR00T).** `GrootRobot` turns each primitive into an instruction such as *"pick the black knight on g8 and place it on f6"*. It then runs the usual VLA loop: send camera frames, proprioception and the instruction to the GR00T inference server, run `action_horizon` steps of the returned action chunk, and repeat until the arm reports the step is done. Two parts are pluggable:

- `RobotIO`: your arm, gripper and camera driver (`get_images`, `get_state`, `apply_action`, `primitive_done`, `home`)
- `PolicyTransport`: `ZmqPolicyTransport` uses the GR00T inference service's request/response pattern (`get_action` over ZMQ with msgpack). Observation and action key names, and the serialisation format, vary between GR00T releases. Check them against the server you deploy and subclass if they differ.

## Running on hardware

1. **Rig:** a tabletop arm (for example SO-101 or Franka) and a fixed camera behind White with the whole board in view. Add a wrist camera if your policy uses one. Pick a piece set that contrasts with both square colours.
2. **Collect demos:** teleoperate pick-and-place episodes for every square, piece type and off-board zone (`graveyard_white`, `graveyard_black`, `reserve`). Label each one with the same instruction template the robot will use (`robot.groot.instruction_template`).
3. **Fine-tune GR00T** on that dataset and start its inference server.
4. **Calibrate:** `robot-chess calibrate` with the board empty. Open `calibration_preview.png` and check that a1 is at the bottom-left. If it isn't, re-run with `--rotate 1..3`.
5. **Play:** set `robot.groot.robot_io` in `configs/groot.yaml` to your driver class, then:

```bash
robot-chess -c configs/groot.yaml play
```

## Layout

```
src/robot_chess/
  engine.py            Stockfish / built-in search
  vision/              camera, calibration, occupancy, move detection
  robot/planner.py     move → PickPlace primitives
  robot/groot_client.py  GR00T closed-loop client (ZMQ)
  robot/sim.py         simulated robot with failure injection
  game.py              orchestrator: turns, verify/correct/assist loop
  metrics.py           per-move records, summaries, aggregation
  simulation.py        wiring for sim games and benchmarks
  cli.py               robot-chess {play,calibrate,benchmark}
configs/               default.yaml, groot.yaml
tests/                 engine, planner, vision, sim game, GR00T client
```
