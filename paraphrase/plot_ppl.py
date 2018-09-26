import sys

import matplotlib.pyplot as plt

with open('results.csv') as fin:
	data = []
	for line in fin:
		data.append(float(line.rstrip('\n \t').split('\t')[-1]))

data.reverse()

fig = plt.figure(1, figsize=(10,8))
plt.plot(data)
fig.tight_layout()
plt.savefig('ppl.png', dpi=300, bbox_inches='tight')