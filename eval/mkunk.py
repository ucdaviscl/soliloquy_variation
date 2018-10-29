#!/usr/bin/env python3

import sys
import random

syms = {}

with open(sys.argv[1], 'r', encoding='utf-8') as f:
   for line in f:
      toks = line.split()
      syms[toks[0]] = toks[1]

for line in sys.stdin:
   toks = line.split()
   proctoks = [t if t in syms else '<unk>' for t in toks]
   #proctoks = [t if random.random() > 0.02 else '<unk>' for t in toks]
   print(' '.join(proctoks))

