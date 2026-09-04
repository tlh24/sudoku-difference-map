#!/usr/bin/env python3
"""

Is the sudoku difference map chaotic?

Short answer: no, not in the sense of a positive Lyapunov exponent, and this
script is the evidence.  What it has instead is sensitive dependence created
by *discontinuity*, which looks similar from a distance and is why reordering
a floating point operation is enough to send a puzzle down a different path.

The argument has two halves.

1.  Between switching events the map is affine, and it contracts.

	x' = x + P_i(2 PB(x) - x) - PB(x).  The four rule projections PC1..PC4
	return one-hot vectors, so as long as no argmax changes hands they are
	locally *constant* and contribute nothing to the derivative.  PC5 is
	constant on clue cells and the identity elsewhere.  Every entry of the
	cube then evolves independently of every other -- the only coupling is
	through PB, which averages the five copies at the same cube index -- so
	the Jacobian is block diagonal with 729 blocks of 5x5, of just two kinds
	(see jacobian_blocks()).  Both have spectral norm exactly 1, and their
	singular values predict an RMS contraction of about 0.84 per step for a
	random perturbation.  Nothing here can grow.

2.  Exact ties never go away, and each one is a coin flip.

	At the random 0/1 start ~98% of the 324 rule groups are exactly tied.
	That falls off quickly, as the comment in sudoku_dm.py says -- but it
	falls to about 0.3%, not to zero, and 0.3% of 324 is around one tied
	group per puzzle per iteration, forever.  A tied group is a point of
	discontinuity: the two candidates are bitwise equal, so an arbitrarily
	small perturbation decides the winner, and the winner is a one-hot vector,
	so the trajectories jump apart by O(1) in a single step.

The two halves make a testable prediction that separates this from chaos.
Under a positive Lyapunov exponent L, a perturbation eps grows as eps*e^(Lt),
so the time to separate is (1/L)*log(1/eps) -- ten decades of eps would shift
the divergence curves ten log-units apart in time.  Under tie-driven
switching, eps never has to grow at all; it only has to be nonzero when the
next exact tie turns up.  The separation time should then be *independent* of
eps.  Measured over eps from 1e-13 to 1e-3, it is: a median of one step at
every scale, with the pre-flip ratio ||d||/eps pinned in [0.796, 0.864] -- the
same at every eps, which is what a linear regime looks like, and below 1,
which is the contraction of part 1.

Run it with

	python sensitivity_dm.py

which writes images/sensitivity_dm.png.  Needs torch and matplotlib.

"""

import argparse
import os
import sys

import numpy as np
import torch

import sudoku_dm
import sudoku_dm_gpu


def jacobian_blocks():
	"""
	The two 5x5 Jacobian blocks of one difference map step, taken at a point
	where no argmax changes hands.  Index a single entry of the cube and let
	d_k be the perturbation of copy k there; m = mean(d) is what PB does.

		copy 1..4, any cell:  P_k is locally constant, so d_k' = d_k - m
		copy 5, clue cell:    P_5 is constant too,     so d_5' = d_5 - m
		copy 5, other cell:   P_5 is the identity, x5' = x5 + (2PB - x5) - PB
							  collapses to x5' = PB,  so d_5' = m

	:return: (clue-cell block, non-clue-cell block), each 5x5
	"""

	mean = np.ones((5, 5))/5.0

	clue = np.eye(5) - mean
	free = clue.copy()
	free[4, :] = 1/5.0

	return clue, free


def predicted_contraction(n_clues, n=9):
	"""
	The RMS growth factor a random perturbation should see in one step, from
	the block singular values alone: sqrt(mean of the squared singular
	values), mixed over clue and non-clue cells.  This is what part 1 of the
	docstring predicts and what measure_forks() should come back with.
	:param n_clues: number of givens in the puzzle
	:return: float
	"""

	clue, free = jacobian_blocks()
	sc = np.linalg.svd(clue, compute_uv=False)
	sf = np.linalg.svd(free, compute_uv=False)
	f = n_clues/float(n*n)

	return float(np.sqrt(f*(sc**2).mean() + (1-f)*(sf**2).mean()))


def burn_in(dm, boards, steps=200, seed=3):
	"""
	Run the map from the usual random 0/1 start so the state is a generic
	point of the trajectory rather than the wildly degenerate corner it
	starts in.
	:return: (X (B,5,L), clue_mask, clue_onehot)
	"""

	B, L = boards.shape[0], dm.L
	clues = torch.from_numpy(boards.astype(np.int64)).to(dm.device)
	clue_mask, clue_onehot = dm.encode_clues(clues)

	gen = torch.Generator(device=dm.device).manual_seed(seed)
	X = torch.randint(0, 2, (B, 1, L), generator=gen, device=dm.device,
					  dtype=dm.dtype).expand(B, 5, L).contiguous()

	for _ in range(steps):
		dm.step(X, X.mean(dim=1, keepdim=True), clue_mask, clue_onehot)

	return X, clue_mask, clue_onehot


def rule_gaps(dm, X):
	"""
	Within every one of the 4*n*n rule groups, how far the winner is ahead of
	the runner-up.  A gap of exactly zero is a point of discontinuity: the
	projection is genuinely multi-valued there and which way it goes is
	decided by the tie-breaking rule, or by the last bit of the arithmetic.
	:param X: (B, 5, L)
	:return: (B, 4*n*n) of top1-top2
	"""

	B, n, L = X.shape[0], dm.n, dm.L

	PB = X.mean(dim=1, keepdim=True)
	R = (2.0*PB - X)[:, :4]
	g = R.gather(2, dm.groups_flat.unsqueeze(0).expand(B, 4, L)).view(B, 4, n*n, n)
	top2 = g.topk(2, dim=-1).values

	return (top2[..., 0] - top2[..., 1]).reshape(B, -1)


def tie_history(dm, boards, steps=600, seed=3):
	"""
	How the fraction of exactly tied rule groups falls off along a trajectory.
	:return: (list of iteration numbers, list of tied fractions)
	"""

	X, clue_mask, clue_onehot = burn_in(dm, boards, steps=0, seed=seed)

	ts, fracs = [], []
	for t in range(steps+1):
		ts.append(t)
		fracs.append(float((rule_gaps(dm, X) == 0).double().mean()))
		dm.step(X, X.mean(dim=1, keepdim=True), clue_mask, clue_onehot)

	return np.array(ts), np.array(fracs)


def measure_forks(dm, boards, epsilons, steps=40, burn=200, seed=3):
	"""
	Take one trajectory to a generic point, fork it, nudge the copy by eps in
	a random direction, and watch the two separate.

	Puzzles that have already converged, or that happen to be sitting on a tie
	at the fork, are dropped -- the first have nothing left to diverge and the
	second would separate on the very first step for a reason that has nothing
	to do with eps.

	:param epsilons: perturbation sizes to try
	:param steps: iterations to follow the pair for
	:return: dict with the divergence curves, first-flip times, and the
		per-iteration cube difference of one example pair
	"""

	B, L = boards.shape[0], dm.L
	Xb, _, _ = burn_in(dm, boards, steps=burn, seed=seed)

	# keep the unconverged, tie-free puzzles
	PB = Xb.mean(dim=1, keepdim=True)
	unconverged = ~dm.check(dm.decode(PB.squeeze(1)))
	tie_free = (rule_gaps(dm, Xb) > 0).all(dim=1)
	sel = (unconverged & tie_free).nonzero(as_tuple=True)[0]

	Xs = Xb[sel]
	M = len(sel)
	pair_boards = np.repeat(boards[sel.cpu().numpy()], 2, axis=0)
	clues = torch.from_numpy(pair_boards.astype(np.int64)).to(dm.device)
	clue_mask, clue_onehot = dm.encode_clues(clues)

	NFRAME = 16   # keep the cube difference for this many pairs, for the figure

	out = {'n_pairs': M, 'n_total': B, 'epsilons': list(epsilons),
		   'curves': [], 'first_flip': [], 'ratios': [], 'example': None}

	for eps in epsilons:
		X = Xs.unsqueeze(1).expand(M, 2, 5, L).contiguous()
		gen = torch.Generator(device=dm.device).manual_seed(11)
		u = torch.randn(M, 5, L, device=dm.device, dtype=dm.dtype, generator=gen)
		u /= u.norm(dim=(1, 2), keepdim=True)
		X[:, 1] += eps*u
		X = X.view(2*M, 5, L)

		dist, flips, frames = [], [], []
		for t in range(steps):
			PB = X.mean(dim=1, keepdim=True)
			win = dm._rule_winners((2.0*PB - X)[:, :4]).view(M, 2, 4, dm.n*dm.n)
			flips.append((win[:, 0] != win[:, 1]).sum(dim=(1, 2)).cpu())

			# the consensus state is what decode() reads, so difference that
			pb = PB.view(M, 2, L)
			frames.append((pb[:, 1] - pb[:, 0]).abs()[:NFRAME].cpu())

			dm.step(X, PB, clue_mask, clue_onehot)
			v = X.view(M, 2, 5, L)
			dist.append((v[:, 1] - v[:, 0]).norm(dim=(1, 2)).cpu())

		dist = torch.stack(dist).numpy()
		flips = torch.stack(flips).numpy()

		# when each pair first disagrees about a winner
		never = steps+1
		first = np.where(flips > 0, np.arange(steps)[:, None], never).min(axis=0)
		got = first < never

		# ||d||/eps over the steps strictly before that, i.e. still linear
		linear = [dist[:first[k], k]/eps for k in range(M) if got[k] and first[k] > 0]

		out['curves'].append(dist)
		out['first_flip'].append(np.where(got, first, np.nan))
		out['ratios'].append(np.concatenate(linear) if linear else np.array([np.nan]))

		if out['example'] is None:
			# a pair that stays together for a step or two first, so the
			# picture has a 'before' as well as an 'after'
			cand = [k for k in range(min(NFRAME, M)) if got[k] and 2 <= first[k] <= 6]
			k = cand[0] if cand else 0
			out['example'] = (eps, int(first[k]), torch.stack(frames).numpy()[:, k])

	return out


def cube_image(v, n=9):
	"""
	Lay a 729-vector out as a 27x27 image, losslessly and without projecting
	anything.  The cube is 81 cells by 9 digits, so put the board's 9x9 cells
	on a 9x9 grid and give each cell a 3x3 tile holding its nine digit
	components.  Structure in the cube then shows up as structure in the
	picture: a row rule lights up a row of tiles at one tile position, a block
	rule lights up a 3x3 patch of tiles, and so on.
	:param v: (n^3,) array indexed row*n*n + col*n + digit
	:return: (3n, 3n) array
	"""

	bs = int(np.sqrt(n))

	return np.asarray(v).reshape(n, n, bs, bs).transpose(0, 2, 1, 3).reshape(n*bs, n*bs)


def figure(forks, ties, contraction, fname):
	"""
	Draw the evidence.
	:param forks: the dict from measure_forks()
	:param ties: (iterations, tied fraction) from tie_history()
	:param contraction: predicted RMS growth factor per step
	"""

	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	from matplotlib.colors import LogNorm
	from matplotlib.gridspec import GridSpec

	eps = forks['epsilons']
	cmap = plt.get_cmap('viridis')
	cols = [cmap(i/max(len(eps)-1, 1)) for i in range(len(eps))]

	fig = plt.figure(figsize=(15.5, 7.4))
	gs = GridSpec(2, 6, figure=fig, height_ratios=[1.15, 0.95],
				  hspace=0.30, wspace=0.62,
				  left=0.055, right=0.82, top=0.88, bottom=0.06)

	# --- A: the divergence curves, all on top of each other
	ax = fig.add_subplot(gs[0, 0:2])
	for e, c, d in zip(eps, cols, forks['curves']):
		ax.semilogy(np.arange(1, d.shape[0]+1), np.median(d, axis=1), color=c, lw=1.6,
					label=r'$\epsilon=10^{%d}$' % round(np.log10(e)))
	# what a positive Lyapunov exponent would look like
	t = np.arange(1, forks['curves'][0].shape[0]+1)
	for e, c in zip(eps, cols):
		ax.semilogy(t, np.minimum(e*np.exp(0.6*t), 28.0), color=c, lw=0.9, ls=':')
	ax.set_xlabel('iterations after the fork')
	ax.set_ylabel(r'$\|\delta\|$')
	ax.set_title('A  separation does not depend on $\\epsilon$\n'
				 'dotted: what $\\lambda=0.6$ chaos would look like', fontsize=9.5)
	ax.legend(fontsize=6.5, loc='lower right', ncol=2, framealpha=0.9)
	ax.set_ylim(1e-14, 3e2)
	ax.grid(alpha=0.25, lw=0.5)

	# --- B: separation time vs eps
	ax = fig.add_subplot(gs[0, 2:4])
	med = [np.nanmedian(f) for f in forks['first_flip']]
	hi = [np.nanpercentile(f, 90) for f in forks['first_flip']]
	lo = [np.nanpercentile(f, 10) for f in forks['first_flip']]
	ax.fill_between(eps, lo, hi, color='#4c72b0', alpha=0.2, lw=0)
	ax.semilogx(eps, med, 'o-', color='#4c72b0', lw=1.8, ms=5, label='measured (median, 10-90%)')
	ref = med[-1] + np.log(np.array(eps[-1])/np.array(eps))/0.6
	ax.semilogx(eps, ref, ':', color='#c44e52', lw=1.8,
				label=r'chaos would give $\frac{1}{\lambda}\log(1/\epsilon)$')
	ax.set_xlabel(r'perturbation $\epsilon$')
	ax.set_ylabel('iterations to first winner flip')
	ax.set_title('B  ten decades of $\\epsilon$, same answer', fontsize=9.5)
	ax.legend(fontsize=7, loc='upper left')
	ax.set_ylim(-1.5, max(ref)*1.12)
	ax.grid(alpha=0.25, lw=0.5)

	# --- C: the mechanism -- ties never go away
	ax = fig.add_subplot(gs[0, 4:6])
	ts, fr = ties
	keep = ts > 0
	ax.loglog(ts[keep], fr[keep]*100, color='#55a868', lw=1.8)
	ax.axhline(100.0/324, color='#c44e52', ls='--', lw=1.2)
	ax.text(1.6, 100.0/324*1.35, 'one tied group per puzzle', fontsize=7, color='#c44e52')
	ax.set_xlabel('iteration')
	ax.set_ylabel('% of the 324 rule groups exactly tied')
	ax.set_title('C  the mechanism: exact ties persist', fontsize=9.5)
	ax.grid(alpha=0.25, lw=0.5, which='both')

	# --- D: the 729-d state difference, laid out as 27x27, no projection
	e0, flip, frames = forks['example']
	shown = [0, max(flip-1, 0), flip, flip+1, flip+3, min(flip+10, len(frames)-1)]
	# The magnitudes here span fourteen decades, which on one log scale renders
	# everything before the flip as flat black.  Panel A already carries the
	# magnitude; what these want to show is *where* the map breaks, so scale
	# each frame against the initial uniform level and clip well before the
	# top.  The starting field then sits a third of the way up the colormap
	# and reads as texture, and anything that has flipped saturates.
	base = float(np.median(frames[0][frames[0] > 0]))
	norm = LogNorm(vmin=base*1e-2, vmax=base*1e4)

	axes = []
	for i, k in enumerate(shown):
		ax = fig.add_subplot(gs[1, i])
		im = ax.imshow(cube_image(np.maximum(frames[k], base*1e-2))/base*1.0,
					   norm=LogNorm(vmin=1e-2, vmax=1e4), cmap='viridis',
					   interpolation='nearest')
		for b in (9, 18):
			ax.axhline(b-0.5, color='w', lw=0.6, alpha=0.45)
			ax.axvline(b-0.5, color='w', lw=0.6, alpha=0.45)
		ax.set_xticks([]); ax.set_yticks([])
		ax.set_title('t = {k}{m}'.format(k=k, m='  (flip)' if k == flip else ''),
					 fontsize=8.5, color='#c44e52' if k == flip else 'black')
		axes.append(ax)
	axes[0].set_ylabel(r'$|\delta\,PB|$', fontsize=10)

	cb = fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.07,
					  pad=0.10, aspect=55)
	cb.set_label(r'$|\delta\,PB|$ relative to its initial level, clipped at $10^4$'
				 r'   ($\epsilon = 10^{%d}$)' % round(np.log10(e0)), fontsize=8)
	cb.ax.tick_params(labelsize=7)

	rr = np.concatenate([r[np.isfinite(r)] for r in forks['ratios']])
	fig.text(0.838, 0.475,
			 'D   imaging 729 dimensions\n'
			 '    without projecting them\n\n'
			 '729 = 81 cells x 9 digits, so the cube\n'
			 'tiles as 27x27: a 9x9 grid of cells,\n'
			 'each cell a 3x3 tile of its nine digit\n'
			 'components.  Nothing is discarded.\n\n'
			 'Before the flip the difference is just\n'
			 'speckle at scale eps, dimming ~0.85x a\n'
			 'step -- the contraction of part 1.\n\n'
			 'A flip hands one group to a different\n'
			 'winner, moving a unit between two of\n'
			 'its entries, so exactly two pixels jump\n'
			 'to O(1).  Where they sit names the rule\n'
			 'that fired:\n'
			 '  inside one tile ..... digit per cell\n'
			 '  along a tile row .... digit per row\n'
			 '  down a tile column .. digit per col\n'
			 '  within a 3x3 patch .. digit per box\n'
			 'From there it floods the whole cube.\n\n'
			 'measured ||d\'||/||d|| while linear:\n'
			 '    [%.3f, %.3f]\n'
			 'predicted from the 5x5 blocks:\n'
			 '    %.3f\n\n'
			 'block spectral norm is exactly 1,\n'
			 'so nothing can grow.' % (rr.min(), rr.max(), contraction),
			 fontsize=7.0, va='top', ha='left', family='monospace')

	fig.suptitle('The sudoku difference map is not chaotic: it contracts between steps, '
				 'and jumps at exact ties', fontsize=12)
	fig.savefig(fname, dpi=150, facecolor='white')

	return fname


def main(argv=None):

	parser = argparse.ArgumentParser(
		description="Test whether the sudoku difference map is chaotic.")
	parser.add_argument('input', nargs='?', default='test-xtrm.csv',
		help="csv of puzzles to draw from (default: test-xtrm.csv)")
	parser.add_argument('-n', '--puzzles', type=int, default=1024,
		help="how many puzzles to average over (default: 1024)")
	parser.add_argument('--device', default='cuda', help="torch device (default: cuda)")
	parser.add_argument('-o', '--output', default='images/sensitivity_dm.png',
		help="where to write the figure (default: images/sensitivity_dm.png)")
	args = parser.parse_args(argv)

	with open(args.input) as fid:
		puzzles = [next(fid).strip().split(',')[0] for _ in range(args.puzzles)]
	boards = np.stack([sudoku_dm.parse_puzzle(p) for p in puzzles], 0)

	# float64 so eps can be pushed far below anything float32 could resolve
	dm = sudoku_dm_gpu.DifferenceMap(n=9, device=args.device, dtype=torch.float64)

	clue, free = jacobian_blocks()
	print('Jacobian between switching events, per cube entry (5x5 blocks):')
	for name, M in (('clue cell', clue), ('non-clue cell', free)):
		sv = np.linalg.svd(M, compute_uv=False)
		print('  {n:14s} singular values {s}  spectral norm {m:.6f}'.format(
			n=name, s=np.array2string(np.sort(sv)[::-1], precision=4, suppress_small=True),
			m=sv.max()))
	contraction = predicted_contraction(int((boards != 0).sum(axis=(1, 2)).mean()))
	print('  predicted RMS growth per step: {c:.4f}  (< 1, so perturbations shrink)'.format(
		c=contraction))

	epsilons = [1e-13, 1e-11, 1e-9, 1e-7, 1e-5, 1e-3]
	forks = measure_forks(dm, boards, epsilons)
	print('\nforking {m} of {b} puzzles (unconverged and not already on a tie):'.format(
		m=forks['n_pairs'], b=forks['n_total']))
	print('  {e:>8s}  {t:>18s}  {r:>26s}'.format(
		e='eps', t='median flip step', r="||d||/eps while linear"))
	for e, f, r in zip(epsilons, forks['first_flip'], forks['ratios']):
		r = r[np.isfinite(r)]
		print('  {e:8.0e}  {t:18.1f}  {lo:12.6f} .. {hi:.6f}'.format(
			e=e, t=np.nanmedian(f), lo=r.min(), hi=r.max()))

	ties = tie_history(dm, boards[:512])
	for t in (0, 1, 10, 100, 600):
		print('  ties at iteration {t:3d}: {f:6.2f}% of the 324 rule groups '
			  '({n:.2f} per puzzle)'.format(t=t, f=ties[1][t]*100, n=ties[1][t]*324))

	if os.path.dirname(args.output):
		os.makedirs(os.path.dirname(args.output), exist_ok=True)
	print('\nwrote', figure(forks, ties, contraction, args.output))

	return 0


if __name__ == "__main__":
	sys.exit(main())
