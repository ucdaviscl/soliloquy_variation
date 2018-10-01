import sys

'''
Given a corpus of lines of pairs of sentences separated tab, and split into
two different files.
'''

filename = sys.argv[1]

out1 = filename + '.1'
out2 = filename + '.2'
with open(filename, 'r') as fin:
	with open(out1, 'w') as fout1:
		with open(out2, 'w') as fout2:
			for line in fin:
				sents = line.rstrip('\n').split('\t')
				if len(sents) != 2:
					print(f'Error: cannot split line:\n{line}')
					continue
				fout1.write(sents[0]+'\n')
				fout2.write(sents[1]+'\n')