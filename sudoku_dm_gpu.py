#!/usr/bin/env python3
"""

Batched GPU port of the difference map sudoku solver in sudoku_dm.py.

Same algorithm -- five constraint sets, the average projection, and the map
x' = x + P_i(2 PB(x) - x) - PB(x) -- but with B puzzles marched in lockstep
so the GPU has something to do.  A single 9x9 puzzle is 5*9*9*9 = 3645 floats,
14.6 kB in float32, and every operation in the iteration is elementwise or a
reduction over 9 contiguous-after-gather values, so the whole thing is memory
bound and wants a wide batch.

Two deliberate departures from the numpy version, both agreed up front:

  * No attempt at bit-parity.  X_0 comes from torch's RNG and ties inside a
	group are broken by torch.argmax, which makes no promise about which of
	several equal maxima it returns on CUDA.  Per sudoku_dm.py's own note,
	~99% of the PC2 groups are tied on the first iteration, so trajectories
	diverge from the CPU solver immediately.  Aggregate solve rates should
	agree; individual iteration counts will not.

  * The convergence check is not lagged.  sudoku_dm.solve() tests a PB_X that
	was computed before the previous update, so it notices a solution one
	iteration after reaching it; here `iterations` is exactly the number of
	difference map updates applied before the board came out valid, i.e. one
	less than the same run would report on the CPU.

Everything below works on the cube flattened to L = n*n*n, indexed
row*n*n + col*n + digit, the same convention as sudoku_dm.board_to_Q().

"""

import argparse
import sys
import time

import numpy as np
import torch


def group_tables(n=9):
	"""
	The four sudoku rules as index tables into the flattened n^3 cube.

	Each rule cuts the cube into n*n disjoint groups of n, and the Euclidean
	projection onto "0/1 with exactly one 1 per group" is an argmax within
	each group (see the comment block above PC1 in sudoku_dm.py).  Since all
	four rules have that shape, they differ only in *which* n entries form a
	group -- so instead of four hand-written index expressions we build four
	(n*n, n) tables of flat indices and run one gather/argmax/scatter over a
	stacked (B, 4, L) tensor.  Row g of table k is the g'th group of rule k.

	Table order matches PC1..PC4:
		0  fix (row, digit),   vary col	   each digit once per row
		1  fix (row, col),     vary digit   each cell holds one digit
		2  fix (block, digit), vary cell	each digit once per block
		3  fix (col, digit),   vary row	   each digit once per column

	:param n: board size, a perfect square
	:return: (4, n*n, n) int64 numpy array of flat cube indices
	"""

	bs = int(np.sqrt(n))
	if bs*bs != n:
		raise ValueError("board size {n} is not a perfect square".format(n=n))

	# F[r, c, d] is the flat index of that entry of the cube
	r, c, d = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing='ij')
	F = r*n*n + c*n + d

	# a group is the last axis after we move the varying index to the end
	G1 = F.transpose(0, 2, 1).reshape(n*n, n)		 # [row, digit] x col
	G2 = F.reshape(n*n, n)							# [row, col]   x digit
	G4 = F.transpose(1, 2, 0).reshape(n*n, n)		 # [col, digit] x row

	# split row into (block row, inner row) and likewise the column, then
	# gather the two inner indices at the end to make the block the group
	Fb = F.reshape(bs, bs, bs, bs, n)				 # [br, ir, bc, ic, d]
	G3 = Fb.transpose(0, 2, 4, 1, 3).reshape(n*n, n)  # [br, bc, digit] x cell

	return np.stack([G1, G2, G3, G4], 0)


def block_cells(n=9):
	"""
	The n cell indices (row*n + col) of each of the n blocks.
	:param n: board size
	:return: (n, n) int64 numpy array
	"""

	bs = int(np.sqrt(n))
	cell = np.arange(n*n).reshape(bs, bs, bs, bs)	 # [br, ir, bc, ic]

	return cell.transpose(0, 2, 1, 3).reshape(n, n)


def check_table(n=9):
	"""
	For the solution check: which row, which column and which block each cell
	belongs to, already scaled into one flat (3, n, n) counter array indexed
	[family, group, digit].  Adding a cell's digit to its entry gives the slot
	to count it in, so a board is checked with one scatter_add.
	:param n: board size
	:return: (3, n*n) int64 numpy array of counter offsets
	"""

	cells = np.arange(n*n)
	rows, cols = cells//n, cells % n

	blocks = np.zeros(n*n, dtype=np.int64)
	for b, members in enumerate(block_cells(n)):
		blocks[members] = b

	family = (np.arange(3)*n*n).reshape(3, 1)

	return np.stack([rows, cols, blocks], 0)*n + family


class DifferenceMap:
	"""
	The device-resident index tables and the iteration itself.  The tables
	depend only on the board size, so they are built once and shared by every
	batch; nothing here holds per-puzzle state.
	"""

	def __init__(self, n=9, device='cuda', dtype=torch.float32):

		self.n = n
		self.L = n*n*n
		self.device = torch.device(device)
		self.dtype = dtype

		G = group_tables(n)
		self.groups = torch.from_numpy(G).to(self.device)				# (4, n*n, n)
		self.groups_flat = self.groups.reshape(4, self.L).contiguous()	# (4, L)
		self.check_base = torch.from_numpy(check_table(n)).to(self.device)  # (3, n*n)

		# a stride-0 scalar, expanded on use as the source of the scatter_adds
		self.one = torch.ones((), device=self.device, dtype=dtype)

	def _rule_winners(self, R):
		"""
		PC1..PC4 at once.  Each rule cuts the cube into n*n disjoint groups of
		n and its projection puts a single 1 at the largest entry of every
		group (see the comment block above PC1 in sudoku_dm.py), so all four
		differ only in which n entries make up a group -- which is exactly what
		the tables from group_tables() say.  Gather each rule's groups out of
		its own copy, take the argmax along the group, and map that back to a
		flat cube index.

		The one-hot result is never materialized: the caller has to add it to
		something, and adding a one-hot is a scatter_add at the winners.

		:param R: (B, 4, L) the reflected iterates, one per rule
		:return: (B, 4, n*n) flat cube index of each group's winner
		"""

		B, n, L = R.shape[0], self.n, self.L

		idx = self.groups_flat.unsqueeze(0).expand(B, 4, L)
		g = R.gather(2, idx).view(B, 4, n*n, n)

		loc = g.argmax(dim=-1, keepdim=True)
		win = self.groups.unsqueeze(0).expand(B, 4, n*n, n).gather(3, loc)

		return win.view(B, 4, n*n)

	def _clue_projection(self, R5, clue_mask, clue_onehot):
		"""
		PC5: every clue cell holds its clue, every other cell is left alone.
		The clue cell has to come out one-hot rather than merely biased -- see
		the PC5 docstring in sudoku_dm.py for what goes wrong otherwise.

		:param R5: (B, L) the reflected iterate of the fifth copy
		:param clue_mask: (B, n*n) bool, True where the puzzle gives a clue
		:param clue_onehot: (B, n*n, n) the clues as one-hot depth vectors
		:return: (B, L)
		"""

		B, n = R5.shape[0], self.n
		cells = R5.reshape(B, n*n, n)

		return torch.where(clue_mask.unsqueeze(-1), clue_onehot, cells).view(B, self.L)

	def step(self, X, PB, clue_mask, clue_onehot):
		"""
		One difference map update, in place: X += P_i(2 PB - X) - PB.

		The whole iteration is memory bound -- every operation is elementwise
		or a reduction over nine values -- so it is written to touch X as few
		times as possible.  In particular the four rule projections are
		one-hot, so instead of building P and doing X + P - PB we subtract PB
		across the board and add the ones back in at the winners.  That costs
		one pass over X instead of three.  It does mean the arithmetic is
		(X - PB) + P rather than (X + P) - PB, which rounds differently in the
		last bit, and that alone sends a puzzle down a different path.  Not
		because errors grow -- between switching events this map contracts,
		see sensitivity_dm.py -- but because a rule group is exactly tied
		about once per puzzle per iteration, and on a tie the last bit is the
		whole decision.  The tie-breaking rule matters for the same reason.

		:param X: (B, 5, L) the five iterates, updated in place
		:param PB: (B, 1, L) the average projection of X, already computed
		:param clue_mask: (B, n*n) bool
		:param clue_onehot: (B, n*n, n)
		"""

		B = X.shape[0]

		R = 2.0*PB - X
		win = self._rule_winners(R[:, :4])
		P5 = self._clue_projection(R[:, 4], clue_mask, clue_onehot)

		X.sub_(PB)
		X[:, :4].scatter_add_(2, win, self.one.expand(B, 4, self.n*self.n))
		X[:, 4].add_(P5)

	def decode(self, PB):
		"""
		The board each averaged cube is pointing at: the deepest entry of each
		cell.  Digits come back 0-based.
		:param PB: (B, L)
		:return: (B, n*n) int64 of 0..n-1
		"""

		return PB.view(-1, self.n*self.n, self.n).argmax(dim=-1)

	def check(self, board):
		"""
		Is each board a complete valid grid?  The same question
		sudoku_dm.check_solution() asks -- a valid grid, not necessarily this
		puzzle's solution.

		Every cell already holds exactly one digit, so all that is left is that
		no row, column or block repeats one.  Count how often each digit turns
		up in each row, column and block, in a single scatter_add over the
		3*n*n counters check_table() lays out, and ask that every count is one.

		:param board: (B, n*n) int64 of 0..n-1
		:return: (B,) bool
		"""

		B, n = board.shape[0], self.n
		slots = 3*n*n

		idx = (self.check_base.unsqueeze(0) + board.unsqueeze(1)).view(B, slots)
		counts = torch.zeros(B, slots, dtype=self.dtype, device=board.device)
		counts.scatter_add_(1, idx, self.one.expand(B, slots))

		return (counts == 1).all(dim=1)

	def encode_clues(self, boards):
		"""
		Turn a batch of clue boards into the mask and one-hot PC5 needs.
		:param boards: (B, n, n) int64 on device, 0 where empty
		:return: (clue_mask (B,n*n) bool, clue_onehot (B,n*n,n) dtype)
		"""

		B, n = boards.shape[0], self.n
		flat = boards.reshape(B, n*n)
		mask = flat != 0

		onehot = torch.zeros(B, n*n, n, dtype=self.dtype, device=flat.device)
		onehot.scatter_(2, (flat-1).clamp(min=0).unsqueeze(-1), 1.0)

		return mask, onehot


def solve_batch(boards, max_iterations=1000, seed=None, device='cuda',
				dtype=torch.float32, check_every=25, compact_at=0.25,
				dm=None, progress=None):
	"""
	Run the difference map on a whole batch of puzzles at once.

	Every puzzle takes the same step at the same time, so the batch is only as
	fast as its slowest member.  Rather than mask the finished ones and keep
	paying for them, we snapshot a puzzle's board the first iteration it comes
	out valid and, once enough of the batch has finished, repack the live
	tensors to drop them.  With max_iterations=1000 and most puzzles landing in
	well under a hundred, that is the difference between paying for the tail
	and paying for the whole batch.

	Bigger batches are not automatically better.  A batch of B puzzles carries
	B*5*n^3 floats, and once that no longer fits in the L2 the iteration falls
	back to reading it from memory every pass; on a 24 GB, 72 MB-L2 4090 the
	measured optimum was around B = 2048 (~7800 puzzles/s over test-xtrm.csv,
	against ~15/s for the numpy solver on 32 cores), with B = 8192 about a
	third slower and B = 65536 worse still.  Tune it for the card.

	:param boards: (B, n, n) int array of clues, 0 where empty
	:param max_iterations: give up on a puzzle after this many updates
	:param seed: seed for the random starting point
	:param device: torch device
	:param dtype: float32 or float64
	:param check_every: iterations between host syncs (a sync is needed to
		decide whether to compact, and costs a pipeline stall)
	:param compact_at: repack once this fraction of the live batch is done
	:param dm: a DifferenceMap to reuse; built if not given
	:param progress: optional callback(iteration, n_live, n_solved)
	:return: (found (B,n,n) int64 cpu, iterations (B,) int64 cpu,
			  solved (B,) bool cpu)
	"""

	boards = np.asarray(boards).astype(np.int64)
	B, n = boards.shape[0], boards.shape[1]

	if dm is None:
		dm = DifferenceMap(n=n, device=device, dtype=dtype)
	dev, L = dm.device, dm.L

	gen = torch.Generator(device=dev)
	if seed is not None:
		gen.manual_seed(int(seed))

	clues = torch.from_numpy(boards).to(dev)
	clue_mask, clue_onehot = dm.encode_clues(clues)

	# X_0 is random 0/1 and all five copies start equal, as in sudoku_dm.solve()
	X0 = torch.randint(0, 2, (B, 1, L), generator=gen, device=dev, dtype=dtype)
	X = X0.expand(B, 5, L).contiguous()

	# results for the puzzles still being worked on; written back on compaction
	live = torch.arange(B, device=dev)
	solved_at = torch.full((B,), -1, dtype=torch.int64, device=dev)
	found = torch.zeros((B, n*n), dtype=torch.int64, device=dev)

	# and the finished ones, indexed by the original puzzle number
	out_at = torch.full((B,), -1, dtype=torch.int64, device=dev)
	out_found = torch.zeros((B, n*n), dtype=torch.int64, device=dev)

	for it in range(max_iterations+1):

		PB = X.mean(dim=1, keepdim=True)
		board = dm.decode(PB.squeeze(1))

		newly = dm.check(board) & (solved_at < 0)
		solved_at = torch.where(newly, it, solved_at)
		found = torch.where(newly.unsqueeze(-1), board, found)

		last = it == max_iterations
		if last or (it % check_every == 0 and it > 0):
			done = solved_at >= 0
			n_done = int(done.sum().item())

			if progress is not None:
				progress(it, live.numel(), B - live.numel() + n_done)

			if last or n_done == live.numel() or n_done >= compact_at*live.numel():
				if last:
					# whatever the unsolved ones are pointing at is their answer
					found = torch.where(done.unsqueeze(-1), found, board)
					done = torch.ones_like(done)

				keep = ~done
				sel = done.nonzero(as_tuple=True)[0]
				out_at[live[sel]] = solved_at[sel]
				out_found[live[sel]] = found[sel]

				if last or not bool(keep.any()):
					break

				live, solved_at, found = live[keep], solved_at[keep], found[keep]
				clue_mask, clue_onehot = clue_mask[keep], clue_onehot[keep]
				X, PB = X[keep].contiguous(), PB[keep]

		dm.step(X, PB, clue_mask, clue_onehot)

	iterations = out_at.clamp(min=0)
	solved = out_at >= 0
	# report the number of updates actually run, as sudoku_dm does
	iterations = torch.where(solved, iterations, torch.full_like(iterations, max_iterations))

	return (out_found.view(B, n, n)+1).cpu(), iterations.cpu(), solved.cpu()


def solve_many(puzzles, max_iterations=1000, seed=0, device='cuda',
			   dtype=torch.float32, batch_size=2048, **kwargs):
	"""
	Solve a list of 81-character puzzle strings, a batch at a time.
	:param puzzles: list of puzzle strings
	:param seed: base seed; batch k is started from seed+k
	:return: (list of solution strings, list of iteration counts, list of bools)
	"""

	import sudoku_dm

	dm = DifferenceMap(n=9, device=device, dtype=dtype)

	out_found, out_iters, out_solved = [], [], []
	for start in range(0, len(puzzles), batch_size):
		chunk = puzzles[start:start+batch_size]
		boards = np.stack([sudoku_dm.parse_puzzle(p) for p in chunk], 0)

		found, iters, solved = solve_batch(
			boards, max_iterations=max_iterations, seed=seed+start,
			device=device, dtype=dtype, dm=dm, **kwargs)

		out_found += [sudoku_dm.board_to_string(b) for b in found.numpy()]
		out_iters += iters.tolist()
		out_solved += [bool(s) for s in solved.tolist()]

	return out_found, out_iters, out_solved


def main(argv=None):

	import sudoku_dm

	parser = argparse.ArgumentParser(
		description="Batched GPU difference map sudoku solver.")
	parser.add_argument('puzzle', nargs='*', default=None,
		help="puzzles as 81 characters each, or - to read one per line from stdin.  "
			 "Omitted, the built-in puzzles are solved.")
	parser.add_argument('-i', '--max-iterations', type=int, default=1000,
		help="give up on a puzzle after this many iterations (default: 1000)")
	parser.add_argument('--device', default='cuda',
		help="torch device (default: cuda)")
	parser.add_argument('--dtype', default='float32', choices=['float32', 'float64'],
		help="precision of the difference map state (default: float32)")
	parser.add_argument('-b', '--batch-size', type=int, default=2048,
		help="puzzles marched in lockstep (default: 2048; see solve_batch on why "
			 "bigger is not better)")
	parser.add_argument('--seed', type=int, default=0,
		help="base seed for the random starting points")
	args = parser.parse_args(argv)

	if args.puzzle == ['-']:
		text = [ln.strip() for ln in sys.stdin if ln.strip()]
	elif args.puzzle:
		text = args.puzzle
	else:
		text = [sudoku_dm.board_to_string(sudoku_dm.PUZZLES[k]) for k in sorted(sudoku_dm.PUZZLES)]

	dtype = torch.float32 if args.dtype == 'float32' else torch.float64

	t0 = time.perf_counter()
	found, iters, solved = solve_many(text, max_iterations=args.max_iterations,
									  seed=args.seed, device=args.device, dtype=dtype,
									  batch_size=args.batch_size)
	if torch.device(args.device).type == 'cuda':
		torch.cuda.synchronize()
	elapsed = time.perf_counter() - t0

	for puzzle, f, k, ok in zip(text, found, iters, solved):
		print('{p}  {f}  {k:5d}  {ok}'.format(p=puzzle, f=f, k=k, ok='solved' if ok else '-'))

	n_ok = sum(solved)
	print('{n} puzzles in {t:.3f}s ({r:.0f}/s), solved {s} ({p:.1f}%), mean iters {m:.1f}'.format(
		n=len(text), t=elapsed, r=len(text)/elapsed if elapsed > 0 else 0.0,
		s=n_ok, p=100.0*n_ok/max(len(text), 1), m=sum(iters)/max(len(text), 1)),
		file=sys.stderr)

	return 0 if n_ok == len(text) else 1


if __name__ == "__main__":
	sys.exit(main())
