import sys
import subprocess
import os
import random

import kenlm

binloc = '/data/kenlm/build/bin/'
expdir = 'test_3'

trainsrc = 'dstc6_user_train_5.txt'
paraphrase = 'dstc6_100_parafst.txt'
devset = 'full_dialog.complete.clean.dev.txt'
genfm = 'dstc6_train_addone_{}.txt'
genfm_eft = 'dstc6_train_cumulative_{}.txt'
modelfm = 'dstc6_kenlm_{}.{}'
modelfm_eft = 'dstc6_kenlm_cuml_{}.{}'

effective_paraphrases = 'effective_paraphrases.txt'

baseline = 8.01981490607579

def evaluate(binmodel):
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

def ppt(l):
	for i in l:
		print('\t'.join([str(x) for x in i]))

def conduct():
	with open(trainsrc) as fin:
		data = fin.read()
	with open(paraphrase) as fin:
		paras = fin.read().split('\n')

	effectives = []

	i=0
	for sent in paras:
		if len(sent) == 0:
			continue
		with open(os.path.join(expdir, genfm.format(i)), 'w') as fout:
			fout.write(data)
			fout.write(sent+'\n')

		with open(os.path.join(expdir, genfm.format(i))) as fin:
			with open(os.path.join(expdir, modelfm.format(i, 'arpa')), 'w') as fout:
				subprocess.run([binloc+'lmplz', '-o', '3', '--discount_fallback', '--skip_symbols'], stdin = fin, stdout = fout)
		subprocess.run([binloc+'build_binary', os.path.join(expdir, modelfm.format(i, 'arpa')), os.path.join(expdir, modelfm.format(i, 'binary'))])

		score = evaluate(os.path.join(expdir, modelfm.format(i, 'binary')))
		if score < baseline:
			print(sent)
			effectives.append(sent)
		i+=1

	with open(os.path.join(expdir, 'effective_paraphrases.txt'), 'w') as fout:
		fout.write('\n'.join(effectives)+'\n')

	print('Finished processing {} paraphrases, {} are effective.'.format(i, len(effectives)))

def conduct_effectives():
	with open(trainsrc) as fin:
		data = fin.read()
	with open(os.path.join(expdir, effective_paraphrases)) as fin:
		paras = fin.read().split('\n')
	paras = list(set([x.rstrip(' ') for x in paras if len(x) > 0]))
	random.seed()
	random.shuffle(paras)

	hist = []
	scores = []
	for i, sent in enumerate(paras):
		hist.append(sent)
		with open(os.path.join(expdir, genfm_eft.format(i)), 'w') as fout:
			fout.write(data)
			fout.write('\n'.join(hist)+'\n')

		with open(os.path.join(expdir, genfm_eft.format(i))) as fin:
			with open(os.path.join(expdir, modelfm_eft.format(i, 'arpa')), 'w') as fout:
				subprocess.run([binloc+'lmplz', '-o', '3', '--discount_fallback', '--skip_symbols'], stdin = fin, stdout = fout)
		subprocess.run([binloc+'build_binary', os.path.join(expdir, modelfm_eft.format(i, 'arpa')), os.path.join(expdir, modelfm_eft.format(i, 'binary'))])

		score = evaluate(os.path.join(expdir, modelfm_eft.format(i, 'binary')))

		scores.append([len(hist), score])

	ppt(scores)


if __name__ == '__main__':
	conduct_effectives()

