# Push-Optimal Sokoban Solver

[![make.fun · sokoban-solver-01-run-20261121](https://make.fun/badge/sokoban-solver-01-run-20261121.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)
[![rank](https://make.fun/badge/sokoban-solver-01-run-20261121/Nestaa404/sokoban-solver-01/rank.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)
[![reward](https://make.fun/badge/sokoban-solver-01-run-20261121/Nestaa404/sokoban-solver-01/reward.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)

A Sokoban solver that answers every level of a set in as few **pushes** as
possible, written for the [make.fun bounty](https://make.fun/bounty/sokoban-solver-01-run-20261121)
"Build the best Sokoban solver" (`sokoban-solver-01-run-20261121`).

## Idea

The generated levels are a one-wide hallway with one to four small rooms off
it, and no box can ever leave its room — so a level's push-optimum is the sum
of its rooms' optima. The solver therefore:

1. **Splits** the level into rooms (each room found by flood fill from its
   door, staying on its side of the hallway).
2. **Solves each room push-optimally** with A\* over pushes: the heuristic is
   the minimum-cost Hungarian assignment of boxes to goals by BFS distance
   (admissible: one push moves one box one square), states are canonicalised
   by the player's reachable region, and search is pruned by static dead
   squares (reverse-push reachability from the goals) plus a 2×2 freeze check.
3. **Stitches** the room solutions together with shortest walks through the
   hallway, then verifies the whole move string by replaying it.

A whole-level push-A\* is kept only as a fallback for levels that do not
split cleanly. The search budget is counted in expanded nodes, never in wall
clock, and the program reads no clock and no randomness: two runs write
byte-identical solutions.

## Results

- Practice set: **1,000,000 / 1,000,000** — all 5 instances solved in exactly
  the push-optimal count (13, 16, 29, 47, 60 pushes), in ~0.7 s total.
- The house baseline (`plain-astar` 1.0.0) scores 200,000 on the same set
  (only the first instance solved).
- 12 further sets drawn from the competition's own generator (60 fresh
  instances): 1,000,000 on every set.

## How to run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
```

`main.ts` is a single self-contained Node 22 program (built-in modules only).
It reads `/input/<id>/level.txt` for every instance and writes
`/output/<id>/solution.txt` (moves in `u`/`d`/`l`/`r`, upper case marking a
push), creating each output directory itself.
