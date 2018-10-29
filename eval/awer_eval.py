#!/usr/bin/env python3

import os 
import sys

# training data for the language model must be divided
# into files named 000.txt, 001.txt, 002.txt, ...
# with each file containing 64 sentences.
# (this number is arbitrary.)
#
# data_path is the path to the directory containing 
# these files.

data_path = sys.argv[1]

# top_n is the number of files to consider

top_n = int(sys.argv[2])

flist = []
txtlist = []

# now create files with increasing amount of
# training data

for i in range(top_n):
  outfname = "000-%03d" % i + '.txt'
  infname = data_path + '/' + "%03d" % i + '.txt'
  flist.append(infname)
  argstr = ' '.join(flist)
  os.system('cat ' + argstr + ' > ' + outfname)
  txtlist.append(outfname)

# create language models using increasing amount of
# training data, and check artificial word error rate

for fname in txtlist:
  #print(fname)
  #os.system('wc -l ' + fname)
  os.system('./mklm.sh ' + fname + ' ' + fname + '.lm')
  os.system('cat dev.txt | ./awer.py -v tr.unigrams -f ' + fname + '.lm')
  
print('\n------------> with automatic variety <-------------\n')

# now add automatically generated variety to the 
# training data, and check artificial word error rate

for i in range(top_n):
  nstr = str((i+1)*64)
  os.system('head -' + nstr + ' 000-009.top1 > head.tmp')
  os.system('cat ' + txtlist[i] + ' head.tmp > lm.tmp')
  os.system('./mklm.sh lm.tmp lm2.tmp')
  os.system('cat dev.txt | ./awer.py -v tr.unigrams -f lm2.tmp')

