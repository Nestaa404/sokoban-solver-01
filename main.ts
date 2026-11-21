// Sokoban solver: push-optimal per room.
//
// Idea (own design): the generated levels are a 1-wide hallway with 1-4
// small rooms (3-4 boxes each) off it, and no box can ever leave its room,
// so a level's push-optimum is the sum of its rooms' optima. The solver
// therefore splits a level into rooms, solves each room push-optimally with
// A* over pushes (Hungarian box->goal assignment heuristic, static dead
// squares, canonical player position, transposition table), then stitches
// the room solutions together with shortest walks through the hallway.
// A whole-level push-A* is kept only as a fallback when a level does not
// split cleanly. The search budget is counted in expanded nodes, never in
// wall-clock time, so two runs produce byte-identical output.
//
// Local run: mkdir -p out && MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Deterministic work budget (expanded nodes), shared across the whole set.
const ROOM_EXPANSIONS = 400_000;
const GLOBAL_EXPANSIONS = 40_000;

// Directions: up, down, left, right. Index arithmetic uses (dx, dy) with x/y
// coordinates so no row can wrap around.
const DIRS: ReadonlyArray<{ dx: number; dy: number; ch: string }> = [
  { dx: 0, dy: -1, ch: 'u' },
  { dx: 0, dy: 1, ch: 'd' },
  { dx: -1, dy: 0, ch: 'l' },
  { dx: 1, dy: 0, ch: 'r' },
];

type Grid = {
  w: number;
  h: number;
  wall: Uint8Array; // 1 = wall (includes out-of-level padding)
  goals: number[]; // sorted cell indices
  goalSet: Uint8Array;
  boxes: number[]; // sorted cell indices
  player: number;
};

function parseLevel(text: string): Grid {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goalSet = new Uint8Array(w * h);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x]!;
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') {
        goalSet[i] = 1;
        goals.push(i);
      }
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  goals.sort((a, b) => a - b);
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goals, goalSet, boxes, player };
}

const xOf = (g: Grid, i: number): number => i % g.w;
const yOf = (g: Grid, i: number): number => Math.floor(i / g.w);
const at = (g: Grid, x: number, y: number): number => y * g.w + x;
const inBounds = (g: Grid, x: number, y: number): boolean =>
  x >= 0 && y >= 0 && x < g.w && y < g.h;

// Breadth-first distances from one source over floor cells (boxes ignored).
// Returns an Int32Array with -1 for unreachable cells.
function bfsDist(g: Grid, src: number, passable: (i: number) => boolean): Int32Array {
  const dist = new Int32Array(g.w * g.h).fill(-1);
  dist[src] = 0;
  const q: number[] = [src];
  for (let head = 0; head < q.length; head++) {
    const c = q[head]!;
    const cx = xOf(g, c);
    const cy = yOf(g, c);
    for (const d of DIRS) {
      const nx = cx + d.dx;
      const ny = cy + d.dy;
      if (!inBounds(g, nx, ny)) continue;
      const n = at(g, nx, ny);
      if (dist[n] !== -1 || !passable(n)) continue;
      dist[n] = dist[c]! + 1;
      q.push(n);
    }
  }
  return dist;
}

// Shortest walk (as a string of u/d/l/r) from `from` to `to` over floor cells
// that are not walls and not in `blocked`. Empty string when from === to,
// null when unreachable. Deterministic neighbour order.
function bfsPath(
  g: Grid,
  from: number,
  to: number,
  blocked: (i: number) => boolean,
): string | null {
  if (from === to) return '';
  const prev = new Int32Array(g.w * g.h).fill(-2);
  const prevCh = new Array<string>(g.w * g.h).fill('');
  prev[from] = -1;
  const q: number[] = [from];
  for (let head = 0; head < q.length; head++) {
    const c = q[head]!;
    const cx = xOf(g, c);
    const cy = yOf(g, c);
    for (const d of DIRS) {
      const nx = cx + d.dx;
      const ny = cy + d.dy;
      if (!inBounds(g, nx, ny)) continue;
      const n = at(g, nx, ny);
      if (prev[n] !== -2 || g.wall[n] === 1 || blocked(n)) continue;
      prev[n] = c;
      prevCh[n] = d.ch;
      if (n === to) {
        let s = '';
        for (let t = n; t !== from; t = prev[t]!) s = prevCh[t]! + s;
        return s;
      }
      q.push(n);
    }
  }
  return null;
}

// Flood fill of the cells the player can reach from `src` without moving a
// box. Returns the set as a Uint8Array plus its smallest cell index.
function reachable(
  g: Grid,
  src: number,
  boxSet: Uint8Array,
  passable: (i: number) => boolean,
): { set: Uint8Array; min: number } {
  const set = new Uint8Array(g.w * g.h);
  set[src] = 1;
  let min = src;
  const q: number[] = [src];
  for (let head = 0; head < q.length; head++) {
    const c = q[head]!;
    const cx = xOf(g, c);
    const cy = yOf(g, c);
    for (const d of DIRS) {
      const nx = cx + d.dx;
      const ny = cy + d.dy;
      if (!inBounds(g, nx, ny)) continue;
      const n = at(g, nx, ny);
      if (set[n] === 1 || g.wall[n] === 1 || boxSet[n] === 1 || !passable(n)) continue;
      set[n] = 1;
      if (n < min) min = n;
      q.push(n);
    }
  }
  return { set, min };
}

// Static dead squares over a box floor: a box on one can never reach a goal,
// even ignoring other boxes and player reachability. Computed by reverse
// pushes from the goals: from a live cell a, the cell a-d is live when the
// player could stand at a-2d after the reverse push (a-d a box cell, a-2d a
// cell the player can stand on, which may be outside the box floor, e.g. a
// hallway cell beside a room door).
function deadSquares(
  g: Grid,
  boxFloor: (i: number) => boolean,
  standFloor: (i: number) => boolean,
  goals: number[],
): Uint8Array {
  const alive = new Uint8Array(g.w * g.h);
  const q: number[] = [];
  for (const gl of goals) {
    alive[gl] = 1;
    q.push(gl);
  }
  for (let head = 0; head < q.length; head++) {
    const a = q[head]!;
    const ax = xOf(g, a);
    const ay = yOf(g, a);
    for (const d of DIRS) {
      const cx = ax - d.dx;
      const cy = ay - d.dy;
      if (!inBounds(g, cx, cy)) continue;
      const c = at(g, cx, cy);
      if (alive[c] === 1 || !boxFloor(c)) continue;
      const px = ax - 2 * d.dx;
      const py = ay - 2 * d.dy;
      if (!inBounds(g, px, py)) continue;
      if (!standFloor(at(g, px, py))) continue;
      alive[c] = 1;
      q.push(c);
    }
  }
  const dead = new Uint8Array(g.w * g.h);
  for (let i = 0; i < g.w * g.h; i++) {
    if (boxFloor(i) && alive[i] === 0) dead[i] = 1;
  }
  return dead;
}

// Minimum-cost assignment of boxes to goals from a distance matrix
// dist[b][gl] (box index x goal index), by enumerating permutations.
// n <= 4 here. Infinity when no full assignment exists.
function hungarianSmall(dist: number[][]): number {
  const n = dist.length;
  if (n === 0) return 0;
  const perm: number[] = [];
  for (let i = 0; i < n; i++) perm.push(i);
  let best = Infinity;
  const rec = (k: number, cost: number): void => {
    if (cost >= best) return;
    if (k === n) {
      best = cost;
      return;
    }
    for (let j = k; j < n; j++) {
      [perm[k], perm[j]] = [perm[j]!, perm[k]!];
      const d = dist[k]![perm[k]!]!;
      if (d !== Infinity) rec(k + 1, cost + d);
      [perm[k], perm[j]] = [perm[j]!, perm[k]!];
    }
  };
  rec(0, 0);
  return best;
}

// Min-cost assignment for larger n (fallback path): classic O(n^3)
// Hungarian on a square cost matrix with Infinity replaced by a big number.
function hungarianBig(cost: number[][]): number {
  const n = cost.length;
  if (n === 0) return 0;
  const BIG = 1e9;
  const a: number[][] = [];
  for (let i = 0; i <= n; i++) {
    a.push(new Array<number>(n + 1).fill(0));
    for (let j = 1; j <= n; j++) {
      a[i]![j] = i === 0 || j === 0 ? 0 : cost[i - 1]![j - 1] === Infinity ? BIG : cost[i - 1]![j - 1]!;
    }
  }
  const u = new Array<number>(n + 1).fill(0);
  const v = new Array<number>(n + 1).fill(0);
  const p = new Array<number>(n + 1).fill(0);
  const way = new Array<number>(n + 1).fill(0);
  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Array<number>(n + 1).fill(Infinity);
    const used = new Array<boolean>(n + 1).fill(false);
    do {
      used[j0] = true;
      const i0 = p[j0]!;
      let delta = Infinity;
      let j1 = 0;
      for (let j = 1; j <= n; j++) {
        if (used[j]) continue;
        const cur = a[i0]![j]! - u[i0]! - v[j]!;
        if (cur < minv[j]!) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j]! < delta) {
          delta = minv[j]!;
          j1 = j;
        }
      }
      for (let j = 0; j <= n; j++) {
        if (used[j]) {
          u[p[j]!]! += delta;
          v[j]! -= delta;
        } else {
          minv[j]! -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do {
      const j1 = way[j0]!;
      p[j0] = p[j1]!;
      j0 = j1;
    } while (j0 !== 0);
  }
  let total = 0;
  for (let j = 1; j <= n; j++) {
    const c = a[p[j]!]![j]!;
    if (c >= BIG) return Infinity;
    total += c;
  }
  return total;
}

type Push = { box: number; dir: number }; // dir = index into DIRS

// Generic push-A* over a floor set. `boxes0`/`goals` are global cell indices,
// `player0` the start cell, `floor` membership, `dead` static dead squares,
// `goalTest` extra acceptance (e.g. the player must be able to leave).
// Returns the list of pushes (box cell before the push, direction) or null.
// The heuristic is the min assignment of boxes to goals by BFS distance
// (each push moves one box one cell, so it is admissible); the search is
// optimal for the state space it explores.
function pushAStar(
  g: Grid,
  floor: (i: number) => boolean,
  pfloor: (i: number) => boolean,
  dead: Uint8Array,
  boxes0: number[],
  goals: number[],
  player0: number,
  maxExpansions: number,
  goalTest: (boxes: number[], reach: Uint8Array) => boolean,
  useBigHungarian: boolean,
  cheapHeuristic = false,
): { pushes: Push[]; expansions: number } | null {
  const n = boxes0.length;
  const sortedGoals = [...goals].sort((a, b) => a - b);
  // distFromGoal[gi][cell]: BFS distance ignoring boxes.
  const distFromGoal: Int32Array[] = sortedGoals.map((gl) => bfsDist(g, gl, floor));
  const heuristic = (boxes: number[]): number => {
    if (n === 0) return 0;
    const mat: number[][] = [];
    for (let b = 0; b < n; b++) {
      const row: number[] = [];
      for (let gi = 0; gi < n; gi++) {
        const d = distFromGoal[gi]![boxes[b]!]!;
        row.push(d === -1 ? Infinity : d);
      }
      mat.push(row);
    }
    // Cheap mode (whole-level fallback): each box's nearest goal, summed.
    // Admissible like the assignment, but far cheaper per node for many boxes.
    if (cheapHeuristic) {
      let total = 0;
      for (const row of mat) {
        const m = Math.min(...row);
        if (m === Infinity) return Infinity;
        total += m;
      }
      return total;
    }
    return useBigHungarian ? hungarianBig(mat) : hungarianSmall(mat);
  };

  // State storage.
  const stBoxes: number[][] = [boxes0.slice().sort((a, b) => a - b)];
  const stPlayer: number[] = [player0];
  const stG: number[] = [0];
  const stParent: number[] = [-1];
  const stPush: Push[] = [{ box: -1, dir: -1 }];
  const best = new Map<string, number>();
  const keyOf = (boxes: number[], playerRep: number): string =>
    boxes.join(',') + '|' + playerRep;
  best.set(keyOf(stBoxes[0]!, player0), 0);

  // Binary heap on (f, -g, seq) for deterministic tie-breaking.
  type HeapEntry = { f: number; g: number; seq: number; s: number };
  const heap: HeapEntry[] = [];
  let seq = 0;
  const less = (a: HeapEntry, b: HeapEntry): boolean =>
    a.f < b.f || (a.f === b.f && (a.g > b.g || (a.g === b.g && a.seq < b.seq)));
  const heapPush = (e: HeapEntry): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const heapPop = (): HeapEntry => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };

  const h0 = heuristic(stBoxes[0]!);
  if (h0 === Infinity) return null;
  heapPush({ f: h0, g: 0, seq: seq++, s: 0 });

  let expansions = 0;
  const boxSet = new Uint8Array(g.w * g.h);
  while (heap.length > 0) {
    const cur = heapPop();
    const s = cur.s;
    const boxes = stBoxes[s]!;
    // Skip stale entries (a better g for this state was recorded since).
    if (best.get(keyOf(boxes, stPlayer[s]!)) !== s) continue;
    // Rebuild the box set and the player's reachable region.
    boxSet.fill(0);
    for (const b of boxes) boxSet[b] = 1;
    const reach = reachable(g, stPlayer[s]!, boxSet, pfloor);
    // Goal test: every box on a goal (sorted compare) + extra condition.
    {
      let solved = true;
      for (let i = 0; i < n; i++) {
        if (boxes[i] !== sortedGoals[i]) {
          solved = false;
          break;
        }
      }
      if (solved && goalTest(boxes, reach.set)) {
        const pushes: Push[] = [];
        for (let t = s; t > 0; t = stParent[t]!) pushes.push(stPush[t]!);
        pushes.reverse();
        return { pushes, expansions };
      }
    }
    if (expansions >= maxExpansions) return null;
    expansions++;
    // Generate pushes.
    for (let bi = 0; bi < n; bi++) {
      const b = boxes[bi]!;
      const bx = xOf(g, b);
      const by = yOf(g, b);
      for (let di = 0; di < 4; di++) {
        const d = DIRS[di]!;
        const nx = bx + d.dx;
        const ny = by + d.dy;
        if (!inBounds(g, nx, ny)) continue;
        const dest = at(g, nx, ny);
        if (!floor(dest) || boxSet[dest] === 1 || dead[dest] === 1) continue;
        const px = bx - d.dx;
        const py = by - d.dy;
        if (!inBounds(g, px, py)) continue;
        const behind = at(g, px, py);
        if (reach.set[behind] !== 1) continue;
        // 2x2 freeze: a 2x2 block of walls/boxes holding a box off-goal.
        let frozen = false;
        for (const [ox, oy] of [
          [0, 0],
          [-1, 0],
          [0, -1],
          [-1, -1],
        ] as const) {
          const cx = bx + ox;
          const cy = by + oy;
          if (!inBounds(g, cx, cy) || !inBounds(g, cx + 1, cy + 1)) continue;
          const cells = [
            at(g, cx, cy),
            at(g, cx + 1, cy),
            at(g, cx, cy + 1),
            at(g, cx + 1, cy + 1),
          ];
          let allBlocked = true;
          let hasBoxOffGoal = false;
          for (const c of cells) {
            if (g.wall[c] === 1 || boxSet[c] === 1) {
              if (boxSet[c] === 1 && g.goalSet[c] !== 1) hasBoxOffGoal = true;
            } else {
              allBlocked = false;
              break;
            }
          }
          if (allBlocked && hasBoxOffGoal) {
            // The pushed box lands inside a frozen 2x2: only dead when the
            // block itself cannot move. Approximate: dead if the box that
            // would sit at dest is part of such a block after the push.
            const after = boxSet.slice();
            after[b] = 0;
            after[dest] = 1;
            let blockAll = true;
            let blockBox = false;
            for (const c of cells) {
              if (g.wall[c] === 1 || after[c] === 1) {
                if (after[c] === 1 && g.goalSet[c] !== 1) blockBox = true;
              } else {
                blockAll = false;
                break;
              }
            }
            if (blockAll && blockBox) {
              frozen = true;
              break;
            }
          }
        }
        if (frozen) continue;
        const nb = boxes.slice();
        nb[bi] = dest;
        nb.sort((a, b2) => a - b2);
        // New player position: where the box was; canonicalise to the
        // smallest reachable cell of the new region. Mutate the box set in
        // place around the flood fill instead of copying it per successor.
        boxSet[b] = 0;
        boxSet[dest] = 1;
        const reach2 = reachable(g, b, boxSet, pfloor);
        boxSet[b] = 1;
        boxSet[dest] = 0;
        const rep = reach2.min;
        const k = keyOf(nb, rep);
        const prev = best.get(k);
        if (prev !== undefined && stG[prev]! <= cur.g + 1) continue;
        const h = heuristic(nb);
        if (h === Infinity) continue;
        const t = stBoxes.length;
        stBoxes.push(nb);
        stPlayer.push(rep);
        stG.push(cur.g + 1);
        stParent.push(s);
        stPush.push({ box: b, dir: di });
        best.set(k, t);
        heapPush({ f: cur.g + 1 + h, g: cur.g + 1, seq: seq++, s: t });
      }
    }
  }
  return null;
}

// Expand a list of pushes into LURD moves, walking between pushes.
// `boxes` is the live global box list (only this room's boxes move here);
// walks avoid walls and the current boxes of this room.
function expandPushes(
  g: Grid,
  pfloor: (i: number) => boolean,
  boxes0: number[],
  player0: number,
  pushes: Push[],
): string | null {
  const boxes = boxes0.slice();
  let player = player0;
  let moves = '';
  for (const p of pushes) {
    const b = p.box;
    const d = DIRS[p.dir]!;
    const behind = at(g, xOf(g, b) - d.dx, yOf(g, b) - d.dy);
    const boxSet = new Uint8Array(g.w * g.h);
    for (const bb of boxes) boxSet[bb] = 1;
    const walk = bfsPath(
      g,
      player,
      behind,
      (i) => boxSet[i] === 1 || !pfloor(i),
    );
    if (walk === null) return null;
    moves += walk;
    moves += d.ch.toUpperCase();
    const bi = boxes.indexOf(b);
    boxes[bi] = at(g, xOf(g, b) + d.dx, yOf(g, b) + d.dy);
    player = b;
  }
  return moves;
}

type Room = {
  door: number;
  entry: number; // hallway cell beside the door; the player starts here
  side: number; // -1 above the hallway, +1 below
  cells: number[];
  floorSet: Uint8Array;
  boxes: number[];
  goals: number[];
};

// Split a level into hallway + rooms. Returns null when the shape is not
// the generated one (then the caller falls back to a whole-level search).
function splitRooms(g: Grid): Room[] | null {
  const hy = yOf(g, g.player);
  const rooms: Room[] = [];
  // A cell can hold both a box and a goal ('*'), so boxes and goals are
  // claimed separately: the check is that no two rooms share a box, and no
  // two rooms share a goal.
  const claimedBox = new Uint8Array(g.w * g.h);
  const claimedGoal = new Uint8Array(g.w * g.h);
  for (const side of [-1, 1] as const) {
    const dy = side;
    const y = hy + dy;
    if (y < 0 || y >= g.h) continue;
    for (let x = 0; x < g.w; x++) {
      const door = at(g, x, y);
      if (g.wall[door] === 1) continue;
      // The door cell must open onto the hallway: the hallway cell beside
      // it (same column, row hy) is floor.
      if (g.wall[at(g, x, hy)] === 1) continue;
      // Flood fill the room, staying on this side of the hallway.
      const cells: number[] = [];
      const seen = new Uint8Array(g.w * g.h);
      const q: number[] = [door];
      seen[door] = 1;
      for (let head = 0; head < q.length; head++) {
        const c = q[head]!;
        cells.push(c);
        const cx = xOf(g, c);
        const cy = yOf(g, c);
        for (const d of DIRS) {
          const nx = cx + d.dx;
          const ny = cy + d.dy;
          if (!inBounds(g, nx, ny)) continue;
          // Never cross into the hallway row or the other side.
          if ((side === -1 && ny >= hy) || (side === 1 && ny <= hy)) continue;
          const n = at(g, nx, ny);
          if (seen[n] === 1 || g.wall[n] === 1) continue;
          seen[n] = 1;
          q.push(n);
        }
      }
      const floorSet = seen;
      const boxes = g.boxes.filter((b) => floorSet[b] === 1);
      const goals = g.goals.filter((gl) => floorSet[gl] === 1);
      if (boxes.length === 0 && goals.length === 0) continue;
      if (boxes.length !== goals.length) return null;
      for (const b of boxes) {
        if (claimedBox[b] === 1) return null;
        claimedBox[b] = 1;
      }
      for (const gl of goals) {
        if (claimedGoal[gl] === 1) return null;
        claimedGoal[gl] = 1;
      }
      rooms.push({ door, entry: at(g, x, hy), side, cells, floorSet, boxes, goals });
    }
  }
  if (rooms.length === 0) return null;
  for (const b of g.boxes) if (claimedBox[b] !== 1) return null;
  for (const gl of g.goals) if (claimedGoal[gl] !== 1) return null;
  return rooms;
}

// Solve one room push-optimally from its door. Returns the room's moves
// (walks + pushes) and the player's final cell, or null.
function solveRoom(
  g: Grid,
  room: Room,
  maxExpansions: number,
): { moves: string; endPlayer: number } | null {
  const floor = (i: number): boolean => room.floorSet[i] === 1;
  // The player may also stand on the hallway cell beside the door: a box
  // sitting on the door itself can only be pushed from there.
  const pfloor = (i: number): boolean => room.floorSet[i] === 1 || i === room.entry;
  const dead = deadSquares(g, floor, pfloor, room.goals);
  for (const b of room.boxes) if (dead[b] === 1) return null;
  // Goal: boxes on goals and the player able to walk back out to the door.
  const goalTest = (_boxes: number[], reach: Uint8Array): boolean =>
    reach[room.door] === 1;
  let res = pushAStar(
    g,
    floor,
    pfloor,
    dead,
    room.boxes,
    room.goals,
    room.entry,
    maxExpansions,
    goalTest,
    false,
  );
  if (res === null) {
    // Retry without the exit condition (a room whose optimal finish traps
    // the player is still better solved than left unsolved).
    res = pushAStar(
      g,
      floor,
      pfloor,
      dead,
      room.boxes,
      room.goals,
      room.entry,
      maxExpansions,
      () => true,
      false,
    );
  }
  if (res === null) return null;
  const moves = expandPushes(g, pfloor, room.boxes, room.entry, res.pushes);
  if (moves === null) return null;
  // Final player cell: replay the pushes to find where the player ends.
  let player = room.entry;
  const boxes = room.boxes.slice();
  for (const p of res.pushes) {
    const d = DIRS[p.dir]!;
    const bi = boxes.indexOf(p.box);
    boxes[bi] = at(g, xOf(g, p.box) + d.dx, yOf(g, p.box) + d.dy);
    player = p.box;
  }
  return { moves, endPlayer: player };
}

// Whole-level fallback push-A* (no decomposition).
function solveGlobal(g: Grid, maxExpansions: number): string | null {
  const floor = (i: number): boolean => g.wall[i] === 0;
  const dead = deadSquares(g, floor, floor, g.goals);
  for (const b of g.boxes) if (dead[b] === 1) return null;
  const res = pushAStar(
    g,
    floor,
    floor,
    dead,
    g.boxes,
    g.goals,
    g.player,
    maxExpansions,
    () => true,
    false,
    true,
  );
  if (res === null) return null;
  return expandPushes(g, floor, g.boxes, g.player, res.pushes);
}

// Replay moves on a level: true when legal and every box ends on a goal.
function replayOk(g: Grid, moves: string): boolean {
  const box = new Uint8Array(g.w * g.h);
  for (const b of g.boxes) box[b] = 1;
  let player = g.player;
  for (const ch of moves) {
    const d = DIRS.find((dd) => dd.ch === ch.toLowerCase());
    if (d === undefined) return false;
    const nx = xOf(g, player) + d.dx;
    const ny = yOf(g, player) + d.dy;
    if (!inBounds(g, nx, ny)) return false;
    const next = at(g, nx, ny);
    if (g.wall[next] === 1) return false;
    if (box[next] === 1) {
      const bx = nx + d.dx;
      const by = ny + d.dy;
      if (!inBounds(g, bx, by)) return false;
      const beyond = at(g, bx, by);
      if (g.wall[beyond] === 1 || box[beyond] === 1) return false;
      box[next] = 0;
      box[beyond] = 1;
    }
    player = next;
  }
  for (let i = 0; i < g.w * g.h; i++) {
    if (box[i] === 1 && g.goalSet[i] !== 1) return false;
  }
  return true;
}

function solve(text: string): string {
  const g = parseLevel(text);
  if (g.player < 0 || g.boxes.length === 0) return '';
  if (g.boxes.length !== g.goals.length) return '';

  const rooms = splitRooms(g);
  if (rooms !== null) {
    // Visit rooms nearest-first from the player's current cell (hallway
    // walks cost steps, never pushes, so any order scores the same).
    const remaining = rooms.slice();
    let player = g.player;
    const liveBoxes = g.boxes.slice();
    let allMoves = '';
    let ok = true;
    while (remaining.length > 0) {
      let bestIdx = 0;
      let bestWalk: string | null = null;
      for (let i = 0; i < remaining.length; i++) {
        const r = remaining[i]!;
        const boxSet = new Uint8Array(g.w * g.h);
        for (const b of liveBoxes) boxSet[b] = 1;
        // Walk to the room's entry cell (hallway beside its door): the
        // room's own moves start with the player standing there.
        const walk = bfsPath(g, player, r.entry, (c) => boxSet[c] === 1);
        if (walk !== null && (bestWalk === null || walk.length < bestWalk.length)) {
          bestWalk = walk;
          bestIdx = i;
        }
      }
      if (bestWalk === null) {
        ok = false;
        break;
      }
      const room = remaining.splice(bestIdx, 1)[0]!;
      const solved = solveRoom(g, room, ROOM_EXPANSIONS);
      if (solved === null) {
        ok = false;
        break;
      }
      allMoves += bestWalk + solved.moves;
      // Update live boxes: this room's boxes are now on its goals.
      for (let i = 0; i < room.boxes.length; i++) {
        const bi = liveBoxes.indexOf(room.boxes[i]!);
        liveBoxes[bi] = room.goals[i]!;
      }
      player = solved.endPlayer;
    }
    if (ok && replayOk(g, allMoves)) return allMoves;
    // Decomposition path failed: fall through to the global search.
  }

  const global = solveGlobal(g, GLOBAL_EXPANSIONS);
  if (global !== null && replayOk(g, global)) return global;
  return '';
}

for (const id of readdirSync(I).sort()) {
  const moves = solve(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
