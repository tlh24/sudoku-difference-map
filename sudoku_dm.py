"""

Create by: jesseclark
On: 10/10/16

Solve sudoku using projection onto sets methods.
http://www.pnas.org/content/104/2/418.full
'Searching with iterated maps'
V. Elser, I. Rankenburg, and P. Thibault
vol. 104 no. 2, 2007

"""

import argparse
import colorsys
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def check_solution(board):
	"""
	Check if the board is a solution
	:param board: nxn board as a numpy array
	:return: bool
	"""

	board = np.asarray(board).astype(int)
	n = board.shape[0]
	blocksize = int(np.sqrt(n))
	complete = list(range(1, n+1))

	# check rows are complete set
	for ii in range(n):
		if sorted(board[ii,:].tolist()) != complete:
			return False

	# check columns are a complete set
	for jj in range(n):
		if sorted(board[:,jj].tolist()) != complete:
			return False

	# now check that each 3x3 sub square is a complete set
	for yy in range(blocksize):
		for xx in range(blocksize):
			# get the sub block
			temp = board[(yy*blocksize):(yy+1)*blocksize, (xx*blocksize):(xx+1)*blocksize]
			if sorted(temp.reshape(temp.size).tolist()) != complete:
				return False

	return True


# The five projections below all act on the same nxnxn cube Q, indexed as
# Q[row, col, digit] -- the convention set by board_to_Q() and Q_to_board() --
# and all of them address it through the flattened index row*n*n + col*n + digit.
#
# PC1..PC4 are the four sudoku rules, and each one cuts the n^3 entries of the
# cube into n^2 disjoint groups of n:
#
#	PC1  fix (row, digit),   vary col	 each digit once per row
#	PC2  fix (row, col),     vary digit	 each cell holds one digit
#	PC3  fix (block, digit), vary cell	 each digit once per 3x3 block
#	PC4  fix (col, digit),   vary row	 each digit once per column
#
# Because the groups of a family partition the cube, the Euclidean projection
# onto "0/1 with exactly one 1 per group" is simply an argmax within each
# group: a 1 at the largest entry, 0 at the other n-1.  That is all each of
# these functions does, so they return hard 0/1 cubes with n^2 ones and are
# idempotent.  PC5 is the odd one out -- the clues constrain only their own
# cells, so its set is not built from a partition of the cube and it has to
# leave the unconstrained cells alone; see its docstring.
#
# A solved sudoku is a point in the intersection of all five sets, and that
# intersection is what the difference map in solve() searches for.
#
# Ties within a group go to its lowest index, np.argmax taking the first of
# any equal maxima.  This used to be argsort(...)[::-1] followed by [0], which
# handed the tie to the highest index instead -- and by way of an unstable
# sort, so it was not a guarantee so much as a habit.  Either rule is a
# perfectly good projection (every tied entry is equidistant, the set is not
# convex and the projection is genuinely multi-valued there), so the only
# thing that changed is which solution a given seed walks towards.  It is not
# a rare corner either: X_0 is random 0/1, so ~99% of the PC2 groups are tied
# at the first iteration.  Once the map starts adding real-valued differences,
# exact ties become rare.


def PC1(Q):
	"""
	First projection: each digit appears exactly once in each row.

	The group is Q[row, :, digit] -- one row and one digit, running over the
	columns -- and the flat index k*n*n + n*seq + i walks it, so despite the
	names it is k that runs over the rows here and i over the digits.

	:param Q: nxnxn numpy array
	:return: nxnxn numpy array
	"""

	n = Q.shape[0]
	P = np.zeros(Q.size)
	seq = np.array(range(0,n))

	for i in range(0,n):
		for k in range(0,n):
			ix = k*n*n + n*seq + i
			temp = Q.reshape(n*n*n)
			P[ix[np.argmax(temp[ix])]] = 1

	return P.reshape(Q.shape)


def RC1(Q):
	"""
	First reflecection
	:param Q: nxnxn numpy array
	:return: nxnxn numpy array
	"""
	return 2*PC1(Q) - Q


def PC2(Q):
	"""
	Second projection: each cell holds exactly one digit.

	The group is Q[row, col, :] -- a single cell, running over the digits --
	walked by the flat index k*n*n + j*n + seq, so here k is the row and j the
	column.  This is the constraint that makes the cube readable as a board at
	all: after it every cell has exactly one candidate left, which is what
	Q_to_board() goes looking for.

	:param Q: nxnxn numpy array
	:return: nxnxn numpy array
	"""

	n = Q.shape[0]
	P = np.zeros(Q.size)
	seq = np.array(range(0,n))

	for j in range(0,n):
		for k in range(0,n):
			ix = k*n*n + j*n + seq
			temp = Q.reshape(n*n*n)
			P[ix[np.argmax(temp[ix])]] = 1

	return P.reshape(Q.shape)


def RC2(Q):
	"""
	Second reflection
	:param Q: nxnxn numpy array
	:return: nxnxn numpy array
	"""
	return 2*PC2(Q) - Q


def PC3(Q):
	"""
	Third projection: each digit appears exactly once in each 3x3 block.

	The group is the nine cells of one block, at one digit.  `mask` holds the
	nine cell offsets of the top-left block in row-major cell numbering
	(0,1,2, n,n+1,n+2, 2n,2n+1,2n+2), and adding the block origin
	i*blocksize + j*n*blocksize slides that stencil over the grid -- so i steps
	across the block columns and j down the block rows.  The work is done on
	the (n*n, n) cell-by-digit view of the cube and reshaped back on the way
	out, which is why P is allocated with that shape.

	:param Q: nxnxn numpy array
	:return: nxnxn numpy array
	"""

	n = Q.shape[0]
	P = np.zeros((n*n,n))
	blocksize = int(np.sqrt(n))
	mask = np.zeros((blocksize*blocksize))

	for i in range(0,blocksize):
		for j in range(0,blocksize):
			mask[i+blocksize*j] = i + n*j

	#P = P.reshape(n*n*n)
	for k in range(0,n):
		for i in range(0, blocksize):
			for j in range(0, blocksize):
				ix = (mask + (i*blocksize + j*n*blocksize)).astype(int)
				temp = Q.reshape(n*n,n)
				P[ix[np.argmax(temp[ix,k])],k] = 1

	return P.reshape(Q.shape)


def RC3(Q):
	return 2*PC3(Q) - Q


def PC4(Q):
	"""
	Fourth projection: each digit appears exactly once in each column.

	The mirror of PC1: the group is Q[:, col, digit], running over the rows,
	walked by the flat index seq*n*n + j*n + i, so j is the column and i the
	digit.

	:param Q: nxnxn numpy array
	:return: nxnxn numpy array
	"""

	n = Q.shape[0]
	P = np.zeros(Q.size)
	seq = np.array(range(0,n))

	for i in range(0,n):
		for j in range(0,n):
			ix = seq*n*n + j*n +i
			temp = Q.reshape(n*n*n)
			P[ix[np.argmax(temp[ix])]] = 1
	return P.reshape(Q.shape)


def RC4(Q):
	return 2*PC4(Q) - Q


def PC5(Q, board):
	"""
	Fifth projection - enforce pre-existing constraints.

	The constraint set here is "every clue cell holds its clue", so the
	projection has to make the clue cell one-hot: zero the whole depth vector
	and then set the clue.  Simply writing a 1 at the clue and leaving the
	other eight components alone -- as this used to do -- is not a projection
	onto that set, it is only a bias towards it.  With the weak version the
	five constraint sets intersect in far more than the puzzle's solution, so
	the map happily settles on some *other* valid 9x9 grid: one that completes
	every row, column and block but disagrees with a clue.  Because these
	puzzles are minimal, each clue is the only thing pinning down one
	unavoidable set, and the grids we landed on were the true solution with a
	single unavoidable set re-permuted -- valid, and wrong in exactly one clue.
	See check_solution(), which tests for a valid grid and not for this puzzle's
	solution, hence the 'solved' vs 'matches_solution' split in benchmark_dm.py.

	Cells with no clue are unconstrained, so they pass through untouched.
	We work on a copy: mutating Q in place also broke RC5, whose 2*PC5(Q)-Q
	saw the already-projected array as its Q and collapsed to PC5(Q).

	:param Q: nxnxn numpy array
	:param board nxn array with 1-9 values
	:return: nxnxn numpy array
	"""

	n = Q.shape[0]
	P = Q.copy()

	for i in range(0,n):
		for j in range(0,n):
			if board[i,j] != 0:
				P[i,j,:] = 0
				P[i,j,board[i,j]-1] = 1
	return P



def RC5(Q, board):
	return 2*PC5(Q, board) - Q



def Q_to_board(Q, reverse=False):
	"""
	Take the nxnxn sudoku array (0's and 1's) and turn it into an
	nxn array with the numbers 1-9
	:param Q: nxnxn sudoku array
	:param reverse: reverse the direction
	:return: the nxn array with vals of 1-9
	"""

	n = Q.shape[0]
	board = np.zeros(Q.shape[:2])

	for i in range(0,n):
		for j in range(0,n):
			if not reverse:
				board[i,j] = np.argmax(Q[i,j,:])+1
			else:
				board[j,i] = np.argmax(Q[j,i,:])+1

	return board


def board_to_Q(board, reverse=False):
	"""
	Take the nxn array with the numbers 1-9 and make the
	nxnxn sudoku array (0's and 1's)
	:param board: nxn sudoku board with vals of 1-9
	:param reverse: reverse the direction
	:return: the nxnxn array with each value vectorized
	"""

	n = board.shape[0]
	Q = np.zeros((n,n,n))

	for i in range(0,n):
		for j in range(0,n):
			if not reverse:
				Q[i,j,int(board[i,j]-1)] = 1
			else:
				Q[j,i,int(board[j,i]-1)] = 1
	return Q


def P_avg(X_i):
	""" average projection
	"""
	X_avg = np.mean(X_i,0)

	return np.stack([X_avg for ind in range(1,6)],0)


def P_i(X_i, board):

	# get the projections
	P_is = {0:PC1, 1:PC2, 2:PC3, 3:PC4, 4:PC5}

	return np.array([P_is[ind](X_i[ind]) if ind != 4 else P_is[ind](X_i[ind], board) for ind in range(0, X_i.shape[0])])


# fonts to try, in order; whichever exists first on this machine wins
FONT_PATHS = [
	"/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
	"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
	"/usr/share/fonts/TTF/DejaVuSans.ttf",
	"/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
	"/System/Library/Fonts/Supplemental/Arial.ttf",
	"/Library/Fonts/Arial.ttf",
	"C:\\Windows\\Fonts\\arial.ttf",
]


def get_font(size):
	"""
	Load a scalable font of the requested size, falling back to whatever
	Pillow ships with if none of the usual system fonts are installed.
	:param size: point size
	:return: a PIL ImageFont
	"""

	for path in FONT_PATHS:
		if os.path.exists(path):
			try:
				return ImageFont.truetype(path, size)
			except OSError:
				pass

	try:
		# Pillow >= 10.1 can scale its built-in font
		return ImageFont.load_default(size)
	except TypeError:
		return ImageFont.load_default()


# time constant of the color decay: a freshly flipped digit washes out toward
# white at an initial rate of 1/MAX_AGE per step, then ever more slowly
MAX_AGE = 128

# ages are tracked out to this many time constants; past it the color is white
# to within a rounding error, so there is nothing left to accumulate
AGE_HORIZON = 8

# the available age color maps; see age_color()
AGE_CMAPS = ('cool', 'hsv', 'hot', 'none')


def age_color(age, max_age=MAX_AGE, cmap='cool'):
	"""
	Map the number of steps since a digit last changed to an RGB color.
	Fresh flips are saturated, old ones wash out to white, so everything
	keeps its contrast against the black background: the value channel is
	pinned at 1.0 and only the hue and the saturation carry the age.
	:param age: steps since the digit last flipped (0 == it just flipped)
	:param max_age: time constant of the decay, in steps
	:param cmap: one of AGE_CMAPS
	:return: an (r,g,b) tuple of ints, 0-255
	"""

	if cmap == 'none' or age is None:
		return (255,255,255)

	# fraction of the cycle that has elapsed: exponential, so the color moves
	# fastest right after a flip (at 1/max_age per step) and then eases off.
	if max_age <= 0:
		t = 1.0
	else:
		t = 1.0 - math.exp(-max(age, 0)/float(max_age))

	if cmap == 'hot':
		# red -> orange -> yellow -> white, no hue wrap-around to confuse things
		hue = 0.16*t
	elif cmap == 'hsv':
		# a full turn of the wheel; pretty, but the hue aliases back to red
		hue = t
	else:
		# 'cool': red -> yellow -> green -> cyan -> blue as the flip recedes
		hue = 0.7*t

	# hold the saturation up for most of the cycle, then fade to white at the end
	sat = (1.0-t)**0.5

	rgb = colorsys.hsv_to_rgb(hue, sat, 1.0)

	return tuple(int(round(255*c)) for c in rgb)


def board_ages(boards, max_age=MAX_AGE):
	"""
	Walk a sequence of boards and record, for every cell of every board, how
	many steps it has been since that digit changed.  Cells that have not
	flipped yet (or not for a long while) sit at the horizon, i.e. white.
	:param boards: sequence of nxn boards
	:param max_age: time constant of the color decay, in steps
	:return: list of nxn int arrays, one per board
	"""

	oldest = AGE_HORIZON*max_age

	ages = []
	age  = None
	prev = None
	for board in boards:
		board = np.asarray(board)
		if prev is None:
			# nothing to compare the first board against
			age = np.full(board.shape, oldest, dtype=int)
		else:
			age = np.minimum(age+1, oldest)
			age[board != prev] = 0
		ages.append(age.copy())
		prev = board

	return ages


def draw_centered(draw, xy, text, font, fill='#ffffff'):
	"""
	Draw text centered on xy.  The 'mm' anchor needs a scalable font, so fall
	back to measuring the string when we are stuck with the bitmap default.
	"""

	try:
		draw.text(xy, text, fill=fill, font=font, anchor='mm')
	except ValueError:
		left, top, right, bottom = draw.textbbox((0,0), text, font=font)
		draw.text((xy[0]-(right-left)/2.0, xy[1]-(bottom-top)/2.0), text, fill=fill, font=font)


def boards_to_images(boards, save_dir='frames',
				   save_prefix='it-', n_out=256, font_size=20, extra=10,
				   max_age=MAX_AGE, cmap='cool'):
	"""
	Write one png per iterate, ready to be turned into a movie or gif, e.g.
		convert -delay 10 -loop 0 frames/6-*.png sudoku_dm.gif
	Each digit is colored by how long ago it last changed; see age_color().
	:param boards: list of nxn boards
	:param save_dir: directory to write into (created if needed)
	:param extra: repeat the final board this many times, to pause on the solution
	:param max_age: time constant, in steps, of a flipped digit's fade to white
	:param cmap: age color map, one of AGE_CMAPS
	:return: list of the file names written
	"""

	# add some to the end when the solution has been found
	boards = list(boards) + [boards[-1] for ind in range(max(extra, 0))] if boards else []

	if not boards:
		return []

	if save_dir:
		os.makedirs(save_dir, exist_ok=True)

	# how many steps since each digit last flipped, per frame
	ages = board_ages(boards, max_age=max_age)

	width = max(3, len(str(len(boards)-1)))
	names = []
	for ind, board in enumerate(boards):
		count = str(ind).zfill(width)
		names.append(board_to_image(board, save_dir, save_prefix+count,
								   n_out, font_size, title=count,
								   ages=ages[ind], max_age=max_age, cmap=cmap))

	return names


def board_to_image(board, save_dir='frames',
				   save_prefix='it-', n_out=256, font_size=20, title=None,
				   ages=None, max_age=MAX_AGE, cmap='cool'):
	"""
	Render a single board to a png.
	:param ages: optional nxn array of steps since each digit last changed;
		without it every digit is drawn white
	:return: the file name written
	"""

	# a blank (black) canvas to draw on
	image = Image.new('RGB', (n_out, n_out), (0,0,0))
	draw  = ImageDraw.Draw(image)
	font  = get_font(font_size)

	if title is not None:
		draw.text((5,5), title, fill='#808080', font=get_font(10))

	# the centers of the 9x9 cells
	step = n_out/10.0
	ints = [step*(k+1) for k in range(9)]

	# faint 3x3 block separators
	for k in range(1,3):
		pos = step*(3*k) + step/2.0
		draw.line([(pos,step/2.0), (pos,n_out-step/2.0)], fill=(48,48,48))
		draw.line([(step/2.0,pos), (n_out-step/2.0,pos)], fill=(48,48,48))

	# loop through each number and draw, colored by how stale it is
	for xx in range(9):
		for yy in range(9):
			numb = int(board[yy,xx])
			age  = None if ages is None else int(ages[yy,xx])
			fill = age_color(age, max_age=max_age, cmap=cmap)
			draw_centered(draw, (ints[xx],ints[yy]), str(numb), font, fill=fill)

	# save the image
	if save_dir:
		os.makedirs(save_dir, exist_ok=True)
	fname = os.path.join(save_dir, save_prefix+'.png')
	image.save(fname, "PNG")

	return fname


PUZZLES = {1:np.array([
			[0, 0, 0, 7, 0, 0, 0, 8, 0],
			[0 ,9 ,0, 0, 0, 3, 1 ,0, 0 ],
			[0, 0, 6, 8, 0, 5, 0, 7, 0 ],
			[0, 2, 0, 6, 0, 0, 0, 4, 9 ],
			[0 ,0, 0, 2, 0, 0, 0, 5, 0 ],
			[0 ,0 ,8, 0, 4, 0, 0, 0, 7 ],
			[0 ,0, 0, 9, 0, 0, 0, 3, 0 ],
			[3 ,7, 0, 0, 0, 0, 0, 0, 6 ],
			[1 ,0, 5, 0, 0, 4 ,0 ,0, 0 ]]
			),
		   2:np.array( [
			[0 ,9 ,0, 0, 8, 0, 0, 4, 0],
			[7 ,0, 0, 3, 0, 9, 0, 0, 8 ],
			[0, 0 ,5 ,0, 0, 0, 3, 0, 0 ],
			[0 ,7, 0, 0, 0, 0, 0, 5, 0 ],
			[8 ,0, 0, 0, 2, 0, 0, 0, 6 ],
			[0 ,1, 0, 0, 0, 0, 0, 2 ,0 ],
			[0, 0, 9, 0, 0, 0, 7, 0, 0 ],
			[6 ,0 ,0, 2, 0, 1, 0, 0 ,5 ],
			[0 ,5 ,0 ,0, 3, 0, 0, 8, 0 ]]
			),
		   3:np.array([
			[4 ,8, 3, 9, 0, 1 ,6, 5, 7],
			[9, 6, 7, 3, 4, 5, 8, 2, 1],
			[2, 5, 1, 8, 7, 0, 4, 9, 3],
			[5 ,4, 8, 0, 3, 2, 9, 7, 0],
			[7 ,2, 9, 5, 6, 4, 1, 3, 8],
			[1 ,3, 6, 7, 0, 8, 2, 4, 5],
			[3, 7, 2, 0, 8, 9, 0, 1, 4],
			[8, 1 ,4, 2, 5, 3, 7, 6 ,9],
			[6, 9, 5 ,4 ,1, 7, 3, 8, 0]]
			),
		   4:np.array([
			[0 ,8, 3, 9, 2, 1 ,6, 5, 7],
			[9, 6, 7, 0, 4, 5, 8, 2, 1],
			[2, 5, 1, 8, 7, 6, 4, 9, 3],
			[5 ,4, 8, 0, 3, 0, 9, 7, 6],
			[0 ,2, 9, 5, 6, 4, 1, 3, 8],
			[1 ,0, 6, 7, 9, 8, 2, 4, 5],
			[3, 7, 2, 0, 8, 9, 5, 1, 4],
			[8, 0 ,4, 2, 5, 3, 7, 6 ,9],
			[6, 9, 5 ,4 ,1, 7, 3, 8, 0]]
			),
		   5:np.array([
			[0, 0, 4, 0, 7, 0, 0, 0, 8 ],
			[0, 0, 5, 0, 0, 6, 0, 0, 0],
			[6, 0, 0, 0, 0, 8, 0, 0, 3],
			[0 ,0 ,0 ,0, 9, 0, 0, 1, 7],
			[0, 0, 0, 0, 2, 0, 0, 0, 5],
			[9, 3, 0, 0, 0, 0, 6, 0, 0],
			[2, 0, 0, 0 ,5 ,0, 0, 0, 1],
			[0, 8, 0, 4, 0, 0, 0, 9, 0],
			[0, 7, 0, 0, 1, 0, 0, 8 ,0]]
			),
		   6:np.array([
			[8, 2, 0, 9, 5, 7, 0, 0, 0],
			[0, 1, 0, 0, 0, 2, 0, 0, 8],
			[0, 0, 0, 0, 0, 4, 2, 0, 0],
			[0 ,7 ,0 ,0, 1, 0, 4, 0, 0],
			[0, 8, 5, 0, 0, 0, 1, 0, 0],
			[0, 0, 1, 0, 9, 0, 0, 6, 0],
			[0, 0, 7, 4 ,0 ,0, 0, 0, 0],
			[9, 0, 0, 6, 0, 0, 0, 5, 0],
			[0, 0, 0, 0, 0, 9, 0, 0 ,3]]
			)}


def get_preset_puzzle(numb=1):
	"""
	Get a preset puzzle to solve
	:param numb: which puzzle to return
	:return: nxn board with 0's where the values are missing
	"""

	return PUZZLES[numb].copy()


def parse_puzzle(text):
	"""
	Parse a puzzle given in the standard 81-character format, read row by row,
	with 0 (or '.') marking a cell with no clue.  Whitespace is ignored, so a
	puzzle pasted as nine lines of nine digits works too.
	:param text: the puzzle string
	:return: 9x9 int board with 0's where the values are missing
	"""

	cells = []
	for c in text:
		if c.isspace():
			continue
		if c.isdigit():
			cells.append(int(c))
		elif c in '._*':
			cells.append(0)
		else:
			raise ValueError("unexpected character {c!r} in puzzle".format(c=c))

	if len(cells) != 81:
		raise ValueError("expected 81 cells, got {n}".format(n=len(cells)))

	return np.array(cells, dtype=int).reshape(9,9)


def board_to_string(board):
	"""
	The inverse of parse_puzzle: an 81-character single-line board.
	"""

	return ''.join(str(int(v)) for v in np.asarray(board).reshape(-1))


def solve(board, max_iterations=1000, seed=None, verbose=True):
	"""
	Run the difference map until the board is solved or we run out of iterations.
	:param board: 9x9 int board with 0's where the values are missing
	:return: (solution board, list of errors, list of iterates, solved bool)
	"""

	rng = np.random.default_rng(seed)

	# get the board size
	n = board.shape[0]

	# init the board
	X_0 = rng.integers(0, 2, (n,n,n)).astype(float)

	# get the X_i's
	X_i = np.stack([X_0 for ind in range(1,6)],0)

	# Average projection operator
	PB_X = P_avg(X_i)

	# store errors and iterates
	errors = []
	iterates = [Q_to_board(PB_X[0])]
	solved = False

	for ii in range(0, max_iterations):

		# check the solution
		if check_solution(Q_to_board(PB_X[0])):
			solved = True
			if verbose:
				print('Solution found at iteration {ii}'.format(ii=ii))
			break

		# difference map update
		# x' = x + P_i(2PB(x) - x) - PB(x)

		# get the average projection
		PB_X = P_avg(X_i)
		# get the difference of projected reflection and average projection
		D_X = P_i(2*PB_X-X_i, board) - PB_X

		# update
		X_i = X_i + D_X

		# error between succesive projections
		error = np.abs(PB_X-P_avg(X_i)).mean()

		errors.append(error)
		iterates.append(Q_to_board(PB_X[0]))
		if verbose:
			print('{ii}\t{error}'.format(ii=ii, error=error))

	# convert from nxnxn to nxn board
	return Q_to_board(PB_X[0]), errors, iterates, solved


def main(argv=None):

	parser = argparse.ArgumentParser(
		description="Solve sudoku with the difference map (Elser, Rankenburg & Thibault 2007).")
	parser.add_argument('puzzle', nargs='?', default=None,
		help="puzzle as 81 characters read row by row, 0 or '.' for an empty cell; "
			 "use - to read it from stdin.  Omitted, one of the built-in puzzles is used.")
	parser.add_argument('-p', '--preset', type=int, default=6, choices=sorted(PUZZLES),
		help="which built-in puzzle to solve when none is given on the command line (default: 6)")
	parser.add_argument('-i', '--max-iterations', type=int, default=1000,
		help="give up after this many difference map iterations (default: 1000)")
	parser.add_argument('-d', '--save-dir', default='frames',
		help="directory for the animation frames (default: frames)")
	parser.add_argument('--prefix', default=None,
		help="file name prefix for the frames (default: the preset number, or 'it-')")
	parser.add_argument('--size', type=int, default=256,
		help="width and height of each frame, in pixels (default: 256)")
	parser.add_argument('--no-images', action='store_true',
		help="solve only; do not write the animation frames")
	parser.add_argument('--cmap', default='cool', choices=AGE_CMAPS,
		help="color map for how long ago a digit flipped; 'none' draws every "
			 "digit white (default: cool)")
	parser.add_argument('--max-age', type=int, default=MAX_AGE,
		help="time constant, in steps, of a flipped digit's fade to white (default: %d)" % MAX_AGE)
	parser.add_argument('--seed', type=int, default=None,
		help="seed for the random starting point, for a reproducible run")
	parser.add_argument('-q', '--quiet', action='store_true',
		help="do not print the per-iteration error")
	args = parser.parse_args(argv)

	# get the puzzle, either from the command line or from the presets
	if args.puzzle is not None:
		text = sys.stdin.read() if args.puzzle == '-' else args.puzzle
		try:
			board = parse_puzzle(text)
		except ValueError as err:
			parser.error(str(err))
		prefix = 'it-'
	else:
		board = get_preset_puzzle(args.preset)
		prefix = str(args.preset)+'-'

	if args.prefix is not None:
		prefix = args.prefix

	print('puzzle:')
	print(board)

	soln, errors, iterates, solved = solve(board, max_iterations=args.max_iterations,
										   seed=args.seed, verbose=not args.quiet)

	print('done after {ii} iterations'.format(ii=len(errors)))
	print('solved: {ok}'.format(ok=solved))
	print(soln.astype(int))
	print(board_to_string(soln))

	if not solved:
		print('no solution found; try more iterations or a different --seed')

	# output images of the solution to make a movie of the process
	if not args.no_images:
		names = boards_to_images(iterates, save_dir=args.save_dir,
								 save_prefix=prefix, n_out=args.size,
								 max_age=args.max_age, cmap=args.cmap)
		if names:
			print('wrote {n} frames: {first} .. {last}'.format(
				n=len(names), first=names[0], last=names[-1]))

	return 0 if solved else 1


if __name__ == "__main__":
	sys.exit(main())
