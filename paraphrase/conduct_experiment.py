import sys
import subprocess
import os
import argparse

import kenlm

agpsr = argparse.ArgumentParser(description = 'Helper function for finding out how much reducing training sample impacts perplexity for language model.')
agpsr.add_argument('action', choices = ['train', 'evaluate'])
agpsr.add_argument('-b', '--bin', type = str, default = '', help = 'Location of KenLM binaries')
agpsr.add_argument('-s', '--source', type = str, default = '', help = 'Source text file for training')
agpsr.add_argument('-d', '--dev', type = str, default = '', help = 'Text file for development testing')
agpsr.add_argument('-p', '--partitions', type = int, default = 100, help = 'Number of partitions to divide the training set')
agpsr.add_argument('-m', '--max_partitions', type = int, default = 100, help = 'Maximum number of the partitions to generate')
agpsr.add_argument('-t', '--gen_train', action = 'store_true', help = 'Generate partitioned training files')
agpsr.add_argument('--gen_dev', type = str, default = '', help = 'Specify development file name if spliting source into training and development sets')
agpsr.add_argument('--dev_ratio', type = float, default = .1, help = 'Percentage of source file to be used for development set (if --gen_dev is specified)')
agpsr.add_argument('-x', '--prefix', type = str, default = 'kenlm', help = 'Prefix for generated kenlm models')
agpsr.add_argument('-f', '--folder', type = str, default = '', help = 'Folder for storing generated files and models')
agpsr.add_argument('-r', '--results' type = str, default = 'results.txt', help = 'File to store evaluation results')
agpsr.add_argument('--discount_fallback', action = 'store_true', help = 'Enable KenLM discount fallback')

def gen_model(binloc, expdir, data, gen_train, trainfn, devfn, modelfn, np, maxp, dcfb):
	klmcmd = [binloc+'lmplz', '-o', '3', '--skip_symbols']
	if dcfb:
		klmcmd.append('--discount_fallback')
	for i in range(1,maxp+1):
		if gen_train:
			divp = int(len(data)*i/np)
			with open(os.path.join(expdir, trainfn.format(i)), 'w') as fout:
				for l in range(0,divp):
					fout.write(data[l]+'\n')

		with open(os.path.join(expdir, trainfn.format(i))) as fin:
			with open(os.path.join(expdir, modelfn.format(i, 'arpa')), 'w') as fout:
				subprocess.run(klmcmd, stdin = fin, stdout = fout)
		subprocess.run([binloc+'build_binary', os.path.join(expdir, modelfn.format(i, 'arpa')), os.path.join(expdir, modelfn.format(i, 'binary'))])

def evaluate(expdir, devfn, modelfn, results, maxp):
	ppls = []
	for i in range(1,maxp+1):
		# with open(devfn) as fin:
		# 	keval = subprocess.run([binloc+'query', '-v', 'summary', 'dstc2_kenlm_{}.binary'.format(i)], stdin = fin)
		model = kenlm.Model(os.path.join(expdir, modelfn.format(i, 'binary')))
		score = 0
		n_words = 0
		with open(devfn) as fin:
			for line in fin:
				fs = model.full_scores(line.rstrip('\n '))
				for w in fs:
					n_words+=1
					score += w[0]
		ppls.append(10**(-score/n_words))

	with open(results, 'w') as fout:
		for i in range(len(ppls)):
			fout.write('{}\t{}\n'.format(i, ppls[i]))

if __name__ == '__main__':
	# binloc = '/data/kenlm/build/bin/'
	# source = 'user_train.txt'
	# gen_train = True
	# gen_dev = False

	# trainfn = 'user_train_{}.txt'
	# devfn = 'user_dev.txt'
	# modelfn = 'user_kenlm_baseline_{}.{}'

	# expdir = 'test_1'

	params = agpsr.parse_args()
	binloc = params.bin
	source = params.source
	trainfn,_,suf = source.rpartition('.')
	if trainfn == '':
		trainfn = suf
	trainfn = trainfn + '_{}.txt'
	gen_train = params.gen_train
	if params.gen_dev != '':
		gen_dev = True
		devfn = params.gen_dev
	else:
		gen_dev = False
		devfn = params.dev
	modelfn = params.prefix
	modelfn = modelfn.rstrip('_') + '_p{}.{}'
	expdir = params.folder
	if expdir == '':
		expdir = '.'

	if gen_train:
		data = []
		with open(source) as fin:
			for line in fin:
				data.append(line.rstrip('\n'))
		
		if gen_dev:
			dev_ratio = params.dev_ratio
			dev_data = data[-int(dev_ratio*len(data)):]
			data = data[:-int(dev_ratio*len(data))]
			
			with open(devfn, 'w') as dout:
				for l in dev_data:
					dout.write(l+'\n')

	if params.action == 'train':
		gen_model(binloc, expdir, data, gen_train, trainfn, devfn, modelfn, params.partitions, params.max_partitions, params.discount_fallback)
	elif params.action == 'evaluate':
		evaluate(expdir, devfn, modelfn, params.results, params.max_partitions)
