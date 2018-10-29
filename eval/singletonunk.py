#!/usr/bin/env python3

import sys

syms = {}

data = []

for line in sys.stdin:
   toks = line.split()
   for t in toks:
      if t not in syms:
         syms[t] = 0
      syms[t] += 1
   data.append(toks)

for toks in data:
   toks = [t if syms[t] > int(sys.argv[1]) else '<unk>' for t in toks]
   print(' '.join(toks))

