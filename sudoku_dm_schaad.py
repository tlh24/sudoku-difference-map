#!/usr/bin/env python3
"""
Line-by-line numpy port of the Sudoku PHP solver in Appendix B of

	J. Schaad, "Modeling the 8-Queens Problem and Sudoku using an Algorithm
	based on Projections onto Nonconvex Sets", MSc thesis, UBC, 2010.
	https://open.library.ubc.ca/soa/cIRcle/collections/ubctheses/24/items/1.0071292

It exists to check the thesis's claim that its solver never met a puzzle it
could not solve, and to say how far sudoku_dm.py is from what the thesis ran.

Run with no arguments it solves the ten Inkala puzzles the thesis reports on
(pp. 68-73) and prints the iteration count next to the one in the thesis.  All
ten agree exactly, and the seven intermediate Escargot grids printed in Fig 6.6
agree too, so this is the thesis's trajectory bit for bit.  What the PHP does,
and where sudoku_dm.py differs:

  * Four constraint sets, not five.  The PHP's fourth set ("the given Sudoku",
	C4 in section 6.3.4) is "clue cells hold their clue AND every other cell is
	a unit vector", i.e. one-digit-per-cell and the clues are a single
	projection, and the concur average is over four replicas.  sudoku_dm.py
	splits these into PC2 (unit vector in every cell) and PC5 (clue cells only,
	identity elsewhere) and averages over five.  On a blank cell PC5's replica
	is not constrained at all, so its update collapses to X_5 <- PB: a fifth of
	the average is the previous average.

  * The start is deterministic: the clue one-hots, with all-zero depth vectors
	in the blank cells (makeunit() of a blank is the zero vector; the thesis
	*text* says the blanks are set to 1, the code does not).  There is no seed
	and one trajectory per puzzle.  sudoku_dm.py starts from a random 0/1 cube.

  * The stopping test is a valid grid that also respects the clues, taken on
	the argmax of the diagonal average right after the update.  sudoku_dm.py
	only asks for a valid grid, and looks one iteration late.

  * Ties go to the lowest index (strict > in unitproj), as np.argmax does.

  * The PHP gives up after 100000 iterations (stuckinloop()).

None of the three deviations is a bug in the sense of a wrong projection, but
they do make sudoku_dm.py a different algorithm from the one in the thesis.
`--compare N` runs both, plus the thesis model from a random start, over the
first N puzzles of test-xtrm.csv with the thesis's own 100000 cap.  For the
first 512 (Sept 2026) the three are indistinguishable:

	arm            <=1000  <=4000  <=8000  <=20000  <=50000  <=100000
	thesis          41.2%   55.5%   62.5%    69.5%    76.2%     78.3%
	thesis-rand     42.0%   53.5%   63.3%    71.1%    79.1%     80.9%
	repo            40.8%   52.9%   58.8%    69.7%    76.4%     80.7%

so the thesis's own solver, bit for bit, gives up on about a fifth of these
puzzles under its own cap.  `--cycle PUZZLE` runs it further and hashes the
full state each step: of the first eight failures, five fall into an exactly
periodic orbit (periods 22 to 128, entered between 1.4k and 42k iterations)
and can never solve, and three do solve, at 132k, 328k and 418k iterations.

Requires numpy.
"""

import argparse
import csv
import sys
import time

import numpy as np


# ---------------------------------------------------------------------------
# the projections (sudokufunctions.php)

def unit_axis(Q, axis):
	"""PHP unitproj along one axis: a 1 at the first maximum, 0 elsewhere."""
	idx = np.argmax(Q, axis=axis)
	P = np.zeros_like(Q)
	np.put_along_axis(P, np.expand_dims(idx, axis), 1.0, axis=axis)
	return P


def columnproj(Q):
	"""fix (col, digit), vary row"""
	return unit_axis(Q, 0)


def rowproj(Q):
	"""fix (row, digit), vary col"""
	return unit_axis(Q, 1)


def meetproj(Q):
	"""fix (block, digit), vary the nine cells; cells ordered ir*3+ic as in PHP's 1+i+3*j"""
	T = Q.reshape(3, 3, 3, 3, 9).transpose(0, 2, 4, 1, 3).reshape(3, 3, 9, 9)
	P = unit_axis(T, 3)
	return P.reshape(3, 3, 9, 3, 3).transpose(0, 3, 1, 4, 2).reshape(9, 9, 9)


def givenproj(Q, clue_mask, clue_onehot):
	"""blank cells to their nearest unit vector, clue cells to the clue"""
	P = unit_axis(Q, 2)
	P[clue_mask] = clue_onehot[clue_mask]
	return P


def valid_grid(m):
	"""isValidSudoku() without its clue test: no repeats in any row, column or block."""
	for k in range(9):
		if len(set(m[k, :])) != 9 or len(set(m[:, k])) != 9:
			return False
	for br in range(3):
		for bc in range(3):
			if len(set(m[3*br:3*br+3, 3*bc:3*bc+3].ravel())) != 9:
				return False
	return True


def parse(text):
	"""81 characters, row by row, 0 or . for a blank"""
	cells = [0 if c in '.0' else int(c) for c in text if not c.isspace()]
	if len(cells) != 81:
		raise ValueError("expected 81 cells, got %d" % len(cells))
	return np.array(cells).reshape(9, 9)


def to_string(m):
	return ''.join(str(int(v)) for v in np.asarray(m).ravel())


# ---------------------------------------------------------------------------
# the iteration (sudoku.php)

def solve(board, max_iterations=100000, model='thesis', init='thesis', order='php',
		  check_clues=True, rng=None, trace=None):
	"""
	Difference map / Douglas-Rachford on the product space, T_i <- P_i(2 PD - T_i) + (T_i - PD).

	:param model: 'thesis' -> four sets: column, row, block, given+unit-per-cell
				  'repo'   -> sudoku_dm.py's five: row, cell, block, column, clue-only
	:param init:  'thesis' -> clue one-hots, zeros in the blank cells
				  'ones'   -> clue one-hots, ones in the blank cells (what the thesis text says)
				  'random' -> random 0/1 cube from rng, as sudoku_dm.solve()
	:param order: 'php'  -> P + (T - PD)   'repo' -> T + (P - PD); same up to rounding
	:param check_clues: stop only on a grid that also respects the clues (the PHP does)
	:param trace: optional list; the decoded board after every update is appended
	:return: (board found, updates applied, solved)
	"""

	board = np.asarray(board)
	clue_mask = board != 0
	clue_onehot = np.zeros((9, 9, 9))
	for i, j in zip(*np.nonzero(clue_mask)):
		clue_onehot[i, j, board[i, j]-1] = 1.0

	if init == 'thesis':
		S = clue_onehot.copy()
	elif init == 'ones':
		S = clue_onehot.copy()
		S[~clue_mask] = 1.0
	else:
		S = rng.integers(0, 2, (9, 9, 9)).astype(float)

	if model == 'thesis':
		projs = [columnproj, rowproj, meetproj,
				 lambda Q: givenproj(Q, clue_mask, clue_onehot)]
	else:
		def clueonly(Q):
			P = Q.copy()
			P[clue_mask] = clue_onehot[clue_mask]
			return P
		projs = [rowproj, lambda Q: unit_axis(Q, 2), meetproj, columnproj, clueonly]
	K = len(projs)

	def diag(T):
		acc = T[0] + T[1]
		for t in T[2:]:
			acc = acc + t
		return acc/K		# /4 is exact, /5 is what np.mean does

	T = [S.copy() for _ in range(K)]
	m = np.argmax(diag(T), axis=2) + 1
	for it in range(1, max_iterations+1):
		PD = diag(T)
		new = []
		for i in range(K):
			P = projs[i](2*PD + (-1*T[i]))
			if order == 'php':
				new.append(P + (T[i] + (-1*PD)))
			else:
				new.append(T[i] + (P - PD))
		T = new
		m = np.argmax(diag(T), axis=2) + 1
		if trace is not None:
			trace.append(m.copy())
		if valid_grid(m) and (not check_clues or (m[clue_mask] == board[clue_mask]).all()):
			return m, it, True

	return m, max_iterations, False


def find_cycle(board, max_iterations=1000000):
	"""
	The thesis's solver with periodic-orbit detection: the full float state is
	hashed after every update, and a repeat means the map is on an exact cycle
	and will never leave it.
	:return: ('solved' | 'periodic' | 'unsolved', iterations, period or None)
	"""

	import hashlib

	board = np.asarray(board)
	clue_mask = board != 0
	clue_onehot = np.zeros((9, 9, 9))
	for i, j in zip(*np.nonzero(clue_mask)):
		clue_onehot[i, j, board[i, j]-1] = 1.0
	projs = [columnproj, rowproj, meetproj, lambda Q: givenproj(Q, clue_mask, clue_onehot)]

	T = [clue_onehot.copy() for _ in range(4)]
	seen = {}
	for it in range(1, max_iterations+1):
		PD = (((T[0]+T[1])+T[2])+T[3])*0.25
		T = [projs[i](2*PD + (-1*T[i])) + (T[i] + (-1*PD)) for i in range(4)]
		m = np.argmax((((T[0]+T[1])+T[2])+T[3])*0.25, axis=2) + 1
		if valid_grid(m) and (m[clue_mask] == board[clue_mask]).all():
			return 'solved', it, None
		h = hashlib.blake2b(b''.join(t.tobytes() for t in T), digest_size=16).digest()
		if h in seen:
			return 'periodic', it, it-seen[h]
		seen[h] = it

	return 'unsolved', max_iterations, None


# ---------------------------------------------------------------------------
# the thesis's own test set: Inkala's ten puzzles, pp. 68-73, with the
# iteration counts printed there.  (The thesis's printed grid for 'labyrinth'
# is a valid grid that does not satisfy that puzzle's clues; the puzzle itself
# has a unique solution and the count agrees.)

INKALA = [
	('AI escargot',           6627, '100007090030020008009600500005300900010080002600004000300000010040000007007000300'),
	('AI killer application',  882, '000000070060010004003400200800003050002900700040080009020060007000100900700008060'),
	('AI lucky diamond',       276, '100500400009030000070008005001000030800600500090007008004020010200800600000001002'),
	('AI worm hole',          4998, '080000001007004020600300700002009000100060008030400000001700600090008005000000040'),
	('AI labyrinth',         12025, '100400800040030009009006050050300000000001600000070002004010900700800004020004080'),
	('AI circles',            2410, '005009700060000020100800006010700004007060030600003200000006040090050100800100002'),
	('AI squadron',           3252, '600000200090001005008030040000002001500600900007090000070003002000400500006070080'),
	('AI honeypot',            208, '100000060000100003005002900009001000700040080030500002500400006008060070070005000'),
	('AI tweezers',            688, '000010004030200000600008090007060005900005080000800400040900100700002040005030007'),
	('AI broken brick',       1598, '400060070000000600030002001700008500010400000020950000000000705009100030003040080'),
]

# Fig 6.6: the Escargot iterate the PHP printed at "Current Counter N"
ESCARGOT_SNAPSHOTS = {
	1000: '168547293534129768729638541475312986913586472682954135396752814241893657857461329',
	2000: '186457293536921748429635541275312984714585632628794175358766419841263867967118326',
	3000: '166457293534921178279638544475312986713785462628294731396875415541293627257146329',
	4000: '156837294734529168829641573475312981913786442682194135398275614241963857567451329',
	5000: '184437296434925168729648541275362941413589752698714735352872414841263657867142329',
	6000: '126857493534129678879633541725361984413985762698274135352798416941263257867142359',
}


def check_inkala():
	print('%-22s %7s %7s' % ('puzzle', 'thesis', 'port'))
	allok = True
	for name, want, puzzle in INKALA:
		trace = [] if name == 'AI escargot' else None
		m, it, ok = solve(parse(puzzle), trace=trace)
		flag = '' if (ok and it == want) else '   <-- differs'
		allok &= not flag
		print('%-22s %7d %7s%s' % (name, want, it if ok else 'failed', flag))
		if trace is not None:
			for n, snap in sorted(ESCARGOT_SNAPSHOTS.items()):
				d = sum(a != b for a, b in zip(to_string(trace[n-1]), snap))
				print('%-22s   Fig 6.6 counter %d: %d of 81 cells differ' % ('', n, d))
	return allok


# ---------------------------------------------------------------------------
# the comparison on test-xtrm.csv

ARMS = {
	'thesis':      dict(model='thesis', init='thesis', order='php',  check_clues=True),
	'thesis-rand': dict(model='thesis', init='random', order='php',  check_clues=True),
	'repo':        dict(model='repo',   init='random', order='repo', check_clues=False),
}


def _run_arm(task):
	idx, puzzle, solution, arm, max_it = task
	b = parse(puzzle)
	clue = b != 0
	t0 = time.perf_counter()
	m, it, ok = solve(b, max_iterations=max_it, rng=np.random.default_rng(idx), **ARMS[arm])
	return {'index': idx, 'arm': arm, 'solved': int(ok), 'matches': int(to_string(m) == solution),
			'iterations': it, 'clue_violations': int((m[clue] != b[clue]).sum()),
			'seconds': round(time.perf_counter()-t0, 3)}


def compare(fname, n, max_it, out, jobs):
	import multiprocessing as mp

	rows = [r for r in csv.reader(open(fname)) if r and r[0].strip()][:n]
	tasks = [(i, r[0], r[1] if len(r) > 1 else '', arm, max_it) for arm in ARMS for i, r in enumerate(rows)]
	fields = ['index', 'arm', 'solved', 'matches', 'iterations', 'clue_violations', 'seconds']
	results = []
	with open(out, 'w', newline='') as fid, mp.Pool(jobs) as pool:
		w = csv.DictWriter(fid, fieldnames=fields)
		w.writeheader()
		for k, res in enumerate(pool.imap_unordered(_run_arm, tasks, chunksize=1)):
			w.writerow(res)
			fid.flush()
			results.append(res)
			if k % 100 == 0:
				print('\r%d of %d' % (k, len(tasks)), end='', file=sys.stderr, flush=True)
	print(file=sys.stderr)
	summarize(results, max_it)


def summarize(results, max_it):
	caps = [c for c in (1000, 4000, 8000, 20000, 50000, 100000) if c <= max_it]
	print('%-12s %5s  %s  %s' % ('arm', 'n', '  '.join('<=%-6d' % c for c in caps), 'solved & wrong clue'))
	for arm in ARMS:
		a = [r for r in results if r['arm'] == arm]
		if not a:
			continue
		it = np.array([int(r['iterations']) for r in a])
		ok = np.array([int(r['solved']) for r in a]) == 1
		bad = int((ok & (np.array([int(r['clue_violations']) for r in a]) > 0)).sum())
		print('%-12s %5d  %s  %d' % (arm, len(a), '  '.join('%5.1f%%  ' % (100.0*((it <= c) & ok).mean()) for c in caps), bad))


def main(argv=None):
	ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
	ap.add_argument('puzzle', nargs='?', help="81-character puzzle to solve the way the thesis did")
	ap.add_argument('-i', '--max-iterations', type=int, default=100000)
	ap.add_argument('--compare', type=int, metavar='N',
		help="run the three arms over the first N puzzles of --input")
	ap.add_argument('--cycle', action='store_true',
		help="solve PUZZLE the thesis's way with periodic-orbit detection")
	ap.add_argument('--input', default='test-xtrm.csv')
	ap.add_argument('-o', '--output', default='compare-schaad.csv')
	ap.add_argument('-j', '--jobs', type=int, default=30)
	args = ap.parse_args(argv)

	if args.compare:
		compare(args.input, args.compare, args.max_iterations, args.output, args.jobs)
		return 0
	if args.puzzle and args.cycle:
		status, it, period = find_cycle(parse(args.puzzle), max_iterations=args.max_iterations)
		print(status, it, 'period %d' % period if period else '')
		return 0 if status == 'solved' else 1
	if args.puzzle:
		m, it, ok = solve(parse(args.puzzle), max_iterations=args.max_iterations)
		print(to_string(m), it, 'solved' if ok else 'not solved')
		return 0 if ok else 1
	return 0 if check_inkala() else 1


if __name__ == '__main__':
	sys.exit(main())
