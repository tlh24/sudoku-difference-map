#!/usr/bin/env python3
"""

Benchmark the difference map solver over a csv of puzzles.

Each line of the input is 'puzzle,solution', both in the 81-character format
(see sudoku_dm.parse_puzzle).  For every puzzle we record how many difference
map iterations were needed, whether the iteration converged, and whether it
converged to the solution that came with the puzzle -- check_solution() only
asks for a valid grid, so the map can settle on a grid that quietly breaks a
few of the clues.

By default the work is spread over all the cores, one puzzle per worker
process.  With --device cuda the batched pytorch solver in sudoku_dm_gpu.py is
used instead: --batch-size puzzles are marched through the difference map in
lockstep, which is a couple of hundred times faster, at the cost of two
columns losing their per-puzzle meaning -- 'seconds' becomes the batch's wall
clock shared out over its puzzles, and 'seed' is the batch's seed, since one
random starting point is drawn for the whole batch at once.

Results are appended (and flushed) as soon as each puzzle -- or, on the gpu,
each batch -- finishes, so the output can be watched from another terminal
with

	tail -f bench-test-xtrm.csv

and ctrl-c stops the run without losing what has been measured; --resume picks
it back up where it left off.

"""

import os

# keep each worker to one thread; we get our parallelism from the processes
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
			 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
	os.environ.setdefault(_var, '1')

import argparse
import contextlib
import csv
import multiprocessing as mp
import signal
import sys
import time

import numpy as np

import sudoku_dm


FIELDS = ['index', 'puzzle', 'solved', 'matches_solution', 'iterations',
		  'clue_violations', 'seconds', 'seed', 'found']


def read_puzzles(fname, start=0, limit=None):
	"""
	Read the 'puzzle,solution' lines of the input csv.
	:return: list of (index, puzzle string, solution string)
	"""

	out = []
	with open(fname, newline='') as fid:
		for ind, row in enumerate(csv.reader(fid)):
			if not row or not row[0].strip():
				continue
			if ind < start:
				continue
			if limit is not None and len(out) >= limit:
				break
			puzzle = row[0].strip()
			solution = row[1].strip() if len(row) > 1 else ''
			out.append((ind, puzzle, solution))

	return out


def clue_violations(board, found):
	"""
	How many of the given clues the returned grid disagrees with.
	:param board: 9x9 int board of clues, 0 where empty
	:param found: 9x9 solved board
	:return: int
	"""

	board = np.asarray(board).astype(int)
	found = np.asarray(found).astype(int)
	given = board != 0

	return int((found[given] != board[given]).sum())


def run_one(task):
	"""
	Solve a single puzzle.  Runs in a worker process.
	:param task: (index, puzzle string, solution string, max_iterations, seed)
	:return: dict with one row's worth of results
	"""

	ind, puzzle, solution, max_iterations, seed = task

	board = sudoku_dm.parse_puzzle(puzzle)

	t0 = time.perf_counter()
	found, errors, _, solved = sudoku_dm.solve(board, max_iterations=max_iterations,
											   seed=seed, verbose=False)
	elapsed = time.perf_counter() - t0

	found_str = sudoku_dm.board_to_string(found)

	return {'index': ind,
			'puzzle': puzzle,
			'solved': int(bool(solved)),
			'matches_solution': int(bool(solution) and found_str == solution),
			'iterations': len(errors),
			'clue_violations': clue_violations(board, found),
			'seconds': round(elapsed, 4),
			'seed': seed,
			'found': found_str}


def ignore_sigint():
	"""
	Worker initializer: let the parent be the only one to see ctrl-c.
	"""

	signal.signal(signal.SIGINT, signal.SIG_IGN)


def done_indices(fname):
	"""
	The puzzle indices already present in an output csv, for --resume.
	:return: set of ints
	"""

	seen = set()
	if not os.path.exists(fname):
		return seen

	with open(fname, newline='') as fid:
		for row in csv.DictReader(fid):
			try:
				seen.add(int(row['index']))
			except (TypeError, ValueError, KeyError):
				continue

	return seen


def fmt_time(secs):
	"""
	Seconds as h:mm:ss.
	"""

	secs = int(max(secs, 0))

	return '{h}:{m:02d}:{s:02d}'.format(h=secs//3600, m=(secs//60) % 60, s=secs % 60)


def iter_cpu(puzzles, args):
	"""
	Solve the puzzles one per worker process, yielding a result row as each
	one finishes -- so in whatever order they finish, not input order.
	:param puzzles: list of (index, puzzle string, solution string)
	:param args: the parsed command line
	"""

	jobs = args.jobs if args.jobs > 0 else (os.cpu_count() or 1)
	jobs = max(1, min(jobs, len(puzzles)))

	tasks = [(ind, puzzle, solution, args.max_iterations, args.seed+ind)
			 for ind, puzzle, solution in puzzles]

	print('{n} puzzles, {j} workers, max {i} iterations'.format(
		n=len(tasks), j=jobs, i=args.max_iterations), file=sys.stderr)

	pool = mp.Pool(processes=jobs, initializer=ignore_sigint)
	finished = False
	try:
		for res in pool.imap_unordered(run_one, tasks, chunksize=1):
			yield res
		finished = True
	finally:
		# ctrl-c closes this generator at the yield above, and then the
		# workers have to be killed rather than waited for
		pool.close() if finished else pool.terminate()
		pool.join()


def iter_gpu(puzzles, args):
	"""
	Solve the puzzles --batch-size at a time on the gpu, yielding a whole
	batch's worth of rows once the batch is done.

	A batch runs until every puzzle in it has either converged or hit
	--max-iterations, so it costs about as much wall clock as its worst
	member; sudoku_dm_gpu.solve_batch drops puzzles from the live set as they
	finish, so at least the tail is cheap.

	:param puzzles: list of (index, puzzle string, solution string)
	:param args: the parsed command line
	"""

	import torch

	import sudoku_dm_gpu

	dtype = torch.float32 if args.dtype == 'float32' else torch.float64
	dm = sudoku_dm_gpu.DifferenceMap(n=9, device=args.device, dtype=dtype)

	print('{n} puzzles, {d} in batches of {b}, max {i} iterations'.format(
		n=len(puzzles), d=args.device, b=args.batch_size, i=args.max_iterations),
		file=sys.stderr)

	for start in range(0, len(puzzles), args.batch_size):
		chunk = puzzles[start:start+args.batch_size]
		boards = np.stack([sudoku_dm.parse_puzzle(p) for _, p, _ in chunk], 0)
		seed = args.seed + start

		show = None
		if not args.quiet:
			def show(it, live, done, n=len(chunk), s=start):
				sys.stderr.write(
					'\r  batch at {s}: iteration {it}, {l} of {n} still running   '.format(
						s=s, it=it, l=live, n=n))
				sys.stderr.flush()

		t0 = time.perf_counter()
		found, iters, solved = sudoku_dm_gpu.solve_batch(
			boards, max_iterations=args.max_iterations, seed=seed,
			device=args.device, dtype=dtype, check_every=args.check_every,
			dm=dm, progress=show)
		# the batch's cost, shared out over the puzzles that incurred it
		per = (time.perf_counter()-t0)/len(chunk)

		found, iters, solved = found.numpy(), iters.tolist(), solved.tolist()
		for k, (ind, puzzle, solution) in enumerate(chunk):
			found_str = sudoku_dm.board_to_string(found[k])
			yield {'index': ind,
				   'puzzle': puzzle,
				   'solved': int(bool(solved[k])),
				   'matches_solution': int(bool(solution) and found_str == solution),
				   'iterations': int(iters[k]),
				   'clue_violations': clue_violations(boards[k], found[k]),
				   'seconds': round(per, 6),
				   'seed': seed,
				   'found': found_str}


def main(argv=None):

	parser = argparse.ArgumentParser(
		description="Measure difference map iterations to solve each puzzle of a csv.")
	parser.add_argument('input', nargs='?', default='test-xtrm.csv',
		help="input csv of 'puzzle,solution' lines (default: test-xtrm.csv)")
	parser.add_argument('-o', '--output', default=None,
		help="results csv (default: bench-<input>)")
	parser.add_argument('-j', '--jobs', type=int, default=0,
		help="worker processes, cpu only (default: one per core)")
	parser.add_argument('-i', '--max-iterations', type=int, default=4000,
		help="give up on a puzzle after this many iterations (default: 4000)")
	parser.add_argument('-n', '--limit', type=int, default=None,
		help="only do this many puzzles")
	parser.add_argument('-s', '--start', type=int, default=0,
		help="skip the first this many lines of the input")
	parser.add_argument('--seed', type=int, default=0,
		help="base seed; on the cpu puzzle k is started from seed+k, on the gpu "
			 "the batch starting at k is (default: 0)")
	parser.add_argument('--resume', action='store_true',
		help="append to the output, skipping puzzles already in it")
	parser.add_argument('--device', default='cpu',
		help="'cpu' for the numpy solver on a process pool, or a torch device "
			 "such as 'cuda' for the batched gpu solver (default: cpu)")
	parser.add_argument('-b', '--batch-size', type=int, default=2048,
		help="puzzles marched in lockstep on the gpu (default: 2048)")
	parser.add_argument('--check-every', type=int, default=25,
		help="gpu iterations between host syncs; a sync stalls the pipeline but "
			 "is what lets finished puzzles be dropped (default: 25)")
	parser.add_argument('--dtype', default='float32', choices=['float32', 'float64'],
		help="precision of the gpu difference map state (default: float32)")
	parser.add_argument('-q', '--quiet', action='store_true',
		help="do not print the progress line")
	args = parser.parse_args(argv)

	out_name = args.output
	if out_name is None:
		out_name = 'bench-'+os.path.basename(args.input)

	puzzles = read_puzzles(args.input, start=args.start, limit=args.limit)

	skip = done_indices(out_name) if args.resume else set()
	if skip:
		puzzles = [p for p in puzzles if p[0] not in skip]
		print('resuming: {n} already done, {m} to go'.format(n=len(skip), m=len(puzzles)),
			  file=sys.stderr)

	if not puzzles:
		print('nothing to do', file=sys.stderr)
		return 0

	append = args.resume and os.path.exists(out_name)
	fid = open(out_name, 'a' if append else 'w', newline='')
	writer = csv.DictWriter(fid, fieldnames=FIELDS)
	if not append:
		writer.writeheader()
		fid.flush()

	print('results -> {o}'.format(o=out_name), file=sys.stderr)

	total = len(puzzles)
	count = solved = matched = 0
	iters = 0
	t0 = time.perf_counter()
	interrupted = False

	source = iter_cpu(puzzles, args) if args.device == 'cpu' else iter_gpu(puzzles, args)

	try:
		with contextlib.closing(source):
			for res in source:
				writer.writerow(res)
				# flush every row so the csv can be tail'd from another terminal
				fid.flush()

				count += 1
				solved += res['solved']
				matched += res['matches_solution']
				iters += res['iterations']

				if not args.quiet:
					elapsed = time.perf_counter() - t0
					rate = count/elapsed if elapsed > 0 else 0.0
					eta = (total-count)/rate if rate > 0 else 0.0
					sys.stderr.write(
						'\r{c}/{n}  solved {s:.1f}%  matched {m:.1f}%  '
						'mean iters {it:.0f}  {r:.1f}/s  eta {e}   '.format(
							c=count, n=total, s=100.0*solved/count, m=100.0*matched/count,
							it=iters/count, r=rate, e=fmt_time(eta)))
					sys.stderr.flush()
	except KeyboardInterrupt:
		interrupted = True
	finally:
		fid.close()
		if not args.quiet:
			sys.stderr.write('\n')

	if interrupted:
		print('interrupted', file=sys.stderr)

	elapsed = time.perf_counter() - t0
	print('{c} of {n} puzzles in {t} ({r:.1f}/s)'.format(
		c=count, n=total, t=fmt_time(elapsed),
		r=count/elapsed if elapsed > 0 else 0.0), file=sys.stderr)
	if count:
		print('converged:            {s} ({p:.1f}%)'.format(s=solved, p=100.0*solved/count),
			  file=sys.stderr)
		print('matched the solution: {m} ({p:.1f}%)'.format(m=matched, p=100.0*matched/count),
			  file=sys.stderr)
		print('mean iterations:      {it:.1f}'.format(it=iters/count), file=sys.stderr)
	print('results in {o}'.format(o=out_name), file=sys.stderr)

	return 1 if interrupted else 0


if __name__ == "__main__":
	sys.exit(main())
