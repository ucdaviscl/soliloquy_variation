import sys
import subprocess

import kenlm

binloc = '/data/kenlm/build/bin/'
source = 'dstc2_user_shuffled.txt'

data = []
with open(source) as fin:
	for line in fin:
		data.append(line.rstrip('\n'))

trainfn = 'dstc2_user_train_{}.txt'
devfn = 'dstc2_user_dev.txt'

dev_ratio = .1
dev_data = data[-int(dev_ratio*len(data)):]
data = data[:-int(dev_ratio*len(data))]
# with open(devfn, 'w') as dout:
# 	for l in dev_data:
# 		dout.write(l+'\n')

def gen_model():
	for i in range(1,101):
		divp = int(len(data)*i/100)
		with open(trainfn.format(i), 'w') as fout:
			for l in range(0,divp):
				fout.write(data[l]+'\n')

		with open(trainfn.format(i)) as fin:
			with open('dstc2_kenlm_{}.arpa'.format(i), 'w') as fout:
				subprocess.run([binloc+'lmplz', '-o', '3'], stdin = fin, stdout = fout)
		subprocess.run([binloc+'build_binary', 'dstc2_kenlm_{}.arpa'.format(i), 'dstc2_kenlm_{}.binary'.format(i)])

def eval():
	ppls = []
	for i in range(1,101):
		# with open(devfn) as fin:
		# 	keval = subprocess.run([binloc+'query', '-v', 'summary', 'dstc2_kenlm_{}.binary'.format(i)], stdin = fin)
		model = kenlm.Model('dstc2_kenlm_{}.binary'.format(i))
		score = 0
		n_words = 0
		with open(devfn) as fin:
			for line in fin:
				fs = model.full_scores(line.rstrip('\n '))
				for w in fs:
					n_words+=1
					score += w[0]
		ppls.append(10**(-score/n_words))

	with open('results.csv', 'w') as fout:
		for i in range(len(ppls)):
			fout.write('{}\t{}\n'.format(i, ppls[i]))

if __name__ == '__main__':
	eval()
