#!/usr/bin/env python3

import sys
import tokenizer

v = {}
oov = 0

with open(sys.argv[1], 'r', encoding='utf-8') as f:
    for line in f:
        toks = tokenizer.word_tokenize(line)
        for t in toks:
            if t not in v:
                v[t] = 1

print("Types in training set:", len(v))

with open(sys.argv[2], 'r', encoding='utf-8') as f:
    for line in f:
        toks = tokenizer.word_tokenize(line)
        for t in toks:
            if t not in v:
                oov += 1

print("OOVs:", oov)
