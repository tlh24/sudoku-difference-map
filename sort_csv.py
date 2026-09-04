#!/usr/bin/env python3
"""Sort a CSV in-place by a named column, descending (numeric when possible)."""
import csv, sys

def main():
	fname = sys.argv[1] if len(sys.argv) > 1 else 'bench-test-xtrm.csv'
	col = sys.argv[2] if len(sys.argv) > 2 else 'iterations'

	with open(fname, newline='') as fd:
		reader = csv.reader(fd)
		header = next(reader)
		rows = list(reader)

	i = header.index(col)
	def key(row):
		try:
			return float(row[i])
		except ValueError:
			return float('-inf')
	rows.sort(key=key, reverse=True)

	with open(fname, 'w', newline='') as fd:
		writer = csv.writer(fd)
		writer.writerow(header)
		writer.writerows(rows)
	print(f'sorted {len(rows)} rows of {fname} by {col} (descending)')

if __name__ == '__main__':
	main()
