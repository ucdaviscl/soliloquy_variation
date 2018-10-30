#!/usr/bin/env python3

import os 
import sys

# training data for the language model must be divided
# into files named 000.txt, 001.txt, 002.txt, ...
# with each file containing 64 sentences.
# (this number is arbitrary.)

# data_path is the path to the directory containing 
# these files.
data_path = sys.argv[1]

# top_n is the number of files to consider
top_n = int(sys.argv[2])

# dev_fname is the file with heldout data
dev_fname = sys.argv[3]

# fake_fname is the file with fake utterances
fake_fname = sys.argv[4]

# tmp_dir is the name of a temporary directory
tmp_dir = sys.argv[5]

flist = []
txtlist = []
flen = []

# now create files with increasing amount of
# training data

for i in range(top_n):
  outfname = tmp_dir + "/000-%03d" % i + '.txt'
  infname = data_path + '/' + "%03d" % i + '.txt'
  nlines = sum(1 for line in open(infname, 'r', encoding='utf-8'))
  flist.append(infname)
  flen.append(nlines)
  argstr = ' '.join(flist)
  os.system('cat ' + argstr + ' > ' + outfname)
  txtlist.append(outfname)

# create language models using increasing amount of
# training data, and check artificial word error rate

for fname in txtlist:
  os.system('./oov.py ' + fname + ' ' + dev_fname)
  
print('\n------------> with automatic variety <-------------\n')

# now add automatically generated variety to the 
# training data, and check artificial word error rate

for i in range(top_n):
  nstr = str((i+1)*flen[i])
  os.system('head -' + nstr + ' ' + fake_fname + ' > ' + tmp_dir + '/head.tmp')
  os.system('cat ' + txtlist[i] + ' ' + tmp_dir + '/head.tmp > ' + tmp_dir + '/lm.tmp')
  os.system('./oov.py ' + tmp_dir + '/lm.tmp ' + dev_fname)

