import sys
import subprocess

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
with open(devfn, 'w') as dout:
	for l in dev_data:
		dout.write(l+'\n')

def gen_model():
	for i in range(5,100,5):
		divp = int(len(data)*i/100)
		with open(trainfn.format(i), 'w') as fout:
			for l in range(0,divp):
				fout.write(data[l]+'\n')

		with open(trainfn.format(i)) as fin:
			with open('dstc2_kenlm_{}.arpa'.format(i), 'w') as fout:
				subprocess.run([binloc+'lmplz', '-o', '3'], stdin = fin, stdout = fout)

def eval():
	for i in range(5,100,5):
		subprocess.run([binloc+'build_binary', 'dstc2_kenlm_{}.arpa'.format(i), 'dstc2_kenlm_{}.binary'.format(i)])
		with open(devfn) as fin:
			subprocess.run([binloc+'query', '-v', 'summary', 'dstc2_kenlm_{}.binary'.format(i)], stdin = fin)

if __name__ == '__main__':
	eval()
