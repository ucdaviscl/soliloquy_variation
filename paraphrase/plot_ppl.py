import os
import sys

import matplotlib.pyplot as plt

'''
A helper script for plotting 1-D array of data points. Input should be a file
in which each line contains an x and a y value separated by a tab.
'''
if len(sys.argv) < 2:
	print('Please specify the file name.')
	exit(1)

file = sys.argv[1]
if not os.path.isfile(file):
	print(f'{file} is not a file.')
	exit(1)

with open('results.csv') as fin:
	data = []
	for line in fin:
		data.append(float(line.rstrip('\n \t').split('\t')[-1]))

data.reverse()

fig = plt.figure(1, figsize=(10,8))
plt.plot(data)
fig.tight_layout()
plt.savefig('ppl.png', dpi=300, bbox_inches='tight')