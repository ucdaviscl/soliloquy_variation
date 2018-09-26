import sys
import os
import re

import kenlm


def evalken(binmodel, devset):
	model = kenlm.Model(binmodel)
	score = 0
	n_words = 0
	with open(devset) as fin:
		for line in fin:
			fs = model.full_scores(line.rstrip('\n '))
			for w in fs:
				n_words+=1
				score += w[0]
	return 10**(-score/n_words)

if __name__ == '__main__':
	if len(sys.argv) < 4:
		print('Need to specify folder, prefix and devset, in that order')
		exit(1)
	folder = sys.argv[1].rstrip('/')
	if not os.path.isdir(folder):
		print('Folder {} not found'.format(folder))
		exit(1)
	pref = sys.argv[2]
	dev = sys.argv[3]
	pref = pref.rstrip('_')
	mpref = pref + '_{}.{}'
	mdre = re.compile(pref + '_[0-9]+\.binary')
	numtrials = 0
	for r,ds,fs in os.walk(folder):
		if r.endswith(folder):
			for f in fs:
				if mdre.fullmatch(f):
					numtrials += 1
	print('{} trials found'.format(numtrials))
	with open(mpref.format('eval', 'tsv'), 'w') as fout:
		for i in range(numtrials):
			s = evalken(os.path.join(folder, mpref.format(i+1, 'binary')), dev)
			res = '{}\t{}'.format(i+1, s)
			print(res)
			fout.write(res+'\n')
