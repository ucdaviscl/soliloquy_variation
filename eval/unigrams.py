#!/usr/bin/env python3

import sys
import tokenizer

unigrams = {}
totaltoks = 0
totaltypes = 0

with open(sys.argv[1], 'r', encoding='utf-8') as f:
    for line in f:
        toks = tokenizer.word_tokenize(line)
        for t in toks:
            if t not in unigrams:
                unigrams[t] = 0
                totaltypes += 1
            unigrams[t] += 1
            totaltoks += 1

unilist = sorted(list(unigrams.keys()), key=lambda v: unigrams[v], reverse=True)

for w in unilist:
    print("%s %.5f" % (w, unigrams[w] / totaltoks))

