#!/usr/bin/env python3
"""Solve sudoku via SAT.  Usage:
    sudoku_sat.py <81-char puzzle>     # 0 or . = blank
    sudoku_sat.py -f puzzles.csv       # one puzzle per line; optional ,solution to check
Options: -s SOLVER (default cadical195), -u (check uniqueness), -n N (limit lines),
         -j J (worker processes, default 16 = physical cores; pinned one per core)."""
import os, time, argparse, multiprocessing as mp
from itertools import combinations, islice
from pysat.solvers import Solver

V = lambda r, c, d: r * 81 + c * 9 + d + 1          # d in 0..8 -> var 1..729

def base_cnf():
    cl = []
    for r in range(9):
        for c in range(9):
            cl.append([V(r, c, d) for d in range(9)])           # >=1 value per cell
            cl += [[-V(r, c, a), -V(r, c, b)] for a, b in combinations(range(9), 2)]
    for d in range(9):
        for i in range(9):
            cl.append([V(i, j, d) for j in range(9)])            # row
            cl.append([V(j, i, d) for j in range(9)])            # col
            cl += [[-V(i, a, d), -V(i, b, d)] for a, b in combinations(range(9), 2)]
            cl += [[-V(a, i, d), -V(b, i, d)] for a, b in combinations(range(9), 2)]
        for br in range(0, 9, 3):
            for bc in range(0, 9, 3):
                cells = [(br + i, bc + j) for i in range(3) for j in range(3)]
                cl.append([V(r, c, d) for r, c in cells])        # box
                cl += [[-V(*x, d), -V(*y, d)] for x, y in combinations(cells, 2)]
    return cl

BASE = base_cnf()

def decode(model):
    m = set(l for l in model if l > 0)
    return ''.join(str(d + 1) for r in range(9) for c in range(9)
                   for d in range(9) if V(r, c, d) in m)

def solve(p, s, s2=None, sel=0):
    """s: persistent Solver, clues as assumptions.  s2/sel: optional uniqueness check --
    block the found grid behind selector var `sel`, activated by assuming -sel."""
    asm = [V(i // 9, i % 9, int(ch) - 1) for i, ch in enumerate(p) if ch not in '0.']
    t = time.perf_counter()
    sat = s.solve(assumptions=asm)
    sol = decode(s.get_model()) if sat else None
    uniq = None
    if sat and s2 is not None:
        s2.add_clause([sel] + [-V(i // 9, i % 9, int(ch) - 1) for i, ch in enumerate(sol)])
        uniq = not s2.solve(assumptions=asm + [-sel])
    return sat, sol, time.perf_counter() - t, uniq

# ---- parallel workers: one persistent solver per process, pinned to a physical core ----
W = {}

def winit(q, name, uniq):
    core = q.get()
    try: os.sched_setaffinity(0, {core})
    except OSError: pass
    W['s'], W['name'] = Solver(name=name, bootstrap_with=BASE), name
    W['s2'] = Solver(name=name, bootstrap_with=BASE) if uniq else None
    W['sel'] = 730                      # selector vars 730, 731, ... (base CNF uses 1..729)

def wchunk(chunk):
    """chunk: list of (lineno, puzzle, want). -> (n, tot, mx, [problem strings])"""
    n = mx = 0; tot = 0.0; bad = []
    if W['s2'] is not None:             # fresh per chunk: caps accumulated blocking clauses
        W['s2'].delete(); W['s2'] = Solver(name=W['name'], bootstrap_with=BASE); W['sel'] = 730
    for i, p, want in chunk:
        W['sel'] += 1
        sat, sol, dt, uq = solve(p, W['s'], W['s2'], W['sel'])
        n += 1; tot += dt; mx = max(mx, dt)
        if not sat: bad.append(f'{i}: UNSAT  {p}')
        elif want and sol != want: bad.append(f'{i}: MISMATCH\n  got  {sol}\n  want {want}')
        elif uq is False: bad.append(f'{i}: MULTIPLE SOLUTIONS  {p}')
    return n, tot, mx, bad

def chunks(f, limit, size=2000):
    it = ((i, l.split(',')[0], (l.split(',') + [None])[1])
          for i, l in enumerate((x.strip() for x in f), 1) if l)
    if limit: it = islice(it, limit)
    while (c := list(islice(it, size))): yield c

a = argparse.ArgumentParser()
a.add_argument('puzzle', nargs='?')
a.add_argument('-f', '--file')
a.add_argument('-s', '--solver', default='cadical195')
a.add_argument('-u', '--unique', action='store_true')
a.add_argument('-n', '--num', type=int)
a.add_argument('-j', '--jobs', type=int, default=16)
a = a.parse_args()

def grid(s):
    return '\n'.join(' '.join(s[r * 9 + c] for c in range(9)) for r in range(9))

if __name__ == '__main__':
    if a.puzzle:
        S = Solver(name=a.solver, bootstrap_with=BASE)
        S2 = Solver(name=a.solver, bootstrap_with=BASE) if a.unique else None
        sat, sol, dt, uq = solve(a.puzzle, S, S2, 730)
        print('SAT' if sat else 'UNSAT', f'{dt*1e3:.2f} ms',
              '' if uq is None else ('unique' if uq else 'MULTIPLE SOLUTIONS'))
        if sat: print(sol); print(grid(sol))
    elif a.file:
        cores = sorted({int(open(f'/sys/devices/system/cpu/cpu{c}/topology/'
                                 'thread_siblings_list').read().split(',')[0])
                        for c in os.sched_getaffinity(0)})[:a.jobs]   # physical cores only
        q = mp.Queue()
        for i in range(a.jobs): q.put(cores[i % len(cores)])
        tot = mx = 0.0; n = nbad = 0
        wall = time.perf_counter()
        with mp.Pool(a.jobs, winit, (q, a.solver, a.unique)) as pool, open(a.file) as f:
            for cn, ct, cm, bad in pool.imap_unordered(wchunk, chunks(f, a.num)):
                n += cn; tot += ct; mx = max(mx, cm); nbad += len(bad)
                for b in bad: print(b, flush=True)
        wall = time.perf_counter() - wall
        print(f'{n} puzzles, {nbad} failures | {a.jobs} workers on {len(cores)} cores | '
              f'{wall:.1f} s wall, {n/wall:,.0f} puz/s | cpu {tot:.1f} s, '
              f'{tot/n*1e3:.3f} ms mean, {mx*1e3:.1f} ms max')
    else:
        print(__doc__)
