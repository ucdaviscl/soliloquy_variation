#!/usr/bin/env python3

import sys
import wordvecutil
import tokenizer
import fst
import math
import getopt

import os
import argparse

# we can optionally use nltk to tag the text
# and focus on replacement of specific categories
#
# import nltk

parser = argparse.ArgumentParser(description='Sentence variation')
parser.add_argument('-v', '--vectors', type = str, default = '', help = 'word vectors', required = True)
parser.add_argument('-f', '--fst_lm', type = str, default = '', help = 'fst language model', required = True)
parser.add_argument('-s', '--vocab', type = str, default = '', help = 'LM symbols', required = True)

fst.EPS = '<epsilon>'

class AlterSent:
    def __init__(self, vecfname, lmfname, vocabfname, maxtypes=0):
        self.vecs = wordvecutil.word_vectors(vecfname, maxtypes)
        self.lmfst = fst.load(lmfname)
        self.maxtypes = maxtypes
        self.sent_rescore =	self.sent_rescore_dummy
        with open(vocabfname, 'r', encoding='utf-8') as fp:
            self.syms = set(fp.read().split())

    def sent_rescore_dummy(self, sents):
        nscoredsent = [[sents[i][0], sents[i][0], sents[i][1]] for i in range(len(sents))]
        return nscoredsent

    def fst_alter_sent(self, words, numalts=5, cutoff = 0):
        # with NLTK we could do POS tagging here
        # pos = nltk.pos_tag(text)

        # instead, we just make everything NN
        pos = [(w, 'NN') for w in words]

        altfst = fst.FST()
        altfst.set_initial(0)
        
        for idx, (word, tag) in enumerate(pos):
            # add the word to the lattice
            if word in self.syms:
                altfst.add_transition(idx, idx+1, word, word, 0)
            else:
                altfst.add_transition(idx, idx+1, "<unk>", "<unk>", 0)

            # add word alternatives to the lattice
            if ( tag.startswith('NN') or 
                 tag.startswith('JJ') or tag.startswith('RB') or 
                 tag.startswith('VB') ) and ( not word.startswith("'") and 
                 word not in ['.', ',', ':', '?', '!', '-', '--'] ):
                # 'have', 'has', 'had', 'is', 'are', 'am', \
                #             'was', 'were', 'be', '.', ',', ':', '?', \
                #             '!', '-', '--', 'of'] and \
                nearlist = self.vecs.near(word, 5)

                # check if there are any neighbors at all
                if nearlist == None:
                    continue

                # add each neighbor to the lattice
                for widx, (dist, w) in enumerate(nearlist):
                    if dist > 0.1 and w in self.syms and w != word:
                        altfst.add_transition(idx, idx+1, w, w, (math.log(dist) * -1)/1000)

        # mark the final state in the FST
        altfst.set_final(len(words))

        # rescore the lattice using the language model
        scoredfst = fst.compose(altfst, self.lmfst)

        # get best paths in the rescored lattice
        bestpaths = scoredfst.short_paths(numalts)
 
        altstrings = {}

        # get the strings and weights from the best paths
        for i, path in enumerate(bestpaths):
            path_string = ' '.join(path[1])
            path_weight = path[0]
            if not path_string in altstrings:
                altstrings[path_string] = path_weight

        # sort strings by weight
        scoredstrings = []

        for sent in altstrings:
            score = altstrings[sent]
            scoredstrings.append((score, sent))

        #scoredstrings = self.sent_rescore(scoredstrings)
        scoredstrings.sort()

        if len(scoredstrings) > numalts:
            scoredstrings = scoredstring[:numalts]

        if cutoff > 0:
            scoredstrings = [s for s in scoredstrings if s[0] <= cutoff]
        
        return scoredstrings

def main():
    params = parser.parse_args()

    print('Processing...')
    lv = AlterSent(params.vectors, params.fst_lm, params.vocab, 50000)
    print("Ready")
    try:
        while True:
            line = input()
            if line.rstrip(' \n') == '':
                continue
            print()
            words = tokenizer.word_tokenize(line)
            lines = lv.fst_alter_sent(words,50)

            for i, (score, sent) in enumerate(lines):
                print("%d: [%.3f] %s" % (i, score, sent)) 

            print()
    except EOFError:
        pass

if __name__ == "__main__":
    main()
