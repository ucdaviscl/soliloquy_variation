#!/usr/bin/env python3 

import sys
import tokenizer
import pywrapfst as fst
import math
import functools
import operator
import getopt
import random

class AlterSent:
    def __init__(self, unifname, lmfname, maxtypes=0):
        self.unigrams = []
        with open(unifname, 'r', encoding='utf-8') as fp:
            for line in fp:
                toks = line.split()
                self.unigrams.append((toks[0], float(toks[1])))
        self.unigrams = sorted(self.unigrams, key=lambda t: t[1])
        self.lmfst = fst.Fst.read(lmfname)
        self.maxtypes = maxtypes

    def fst_alter_sent(self, words, numalts=5):
        # create new empty FST
        altfst = fst.Fst()
        altfst.add_state()
        
        for idx, word in enumerate(words):
            # add the word to the lattice or <unk> if out-of-vocabulary
            if word in self.lmfst.input_symbols():
                word_id = self.lmfst.input_symbols().find(word)
                arc = fst.Arc(word_id, word_id, 0, self.get_state_id(idx+1, altfst))
                altfst.add_arc(self.get_state_id(idx, altfst), arc)
            else:
                word_id = self.lmfst.input_symbols().find("<unk>")
                arc = fst.Arc(word_id, word_id, 0, self.get_state_id(idx+1, altfst))
                altfst.add_arc(self.get_state_id(idx, altfst), arc)

            # add word alternatives to the lattice
            nearlist = []
            for i in range(1):
                r = random.random()
                altword = '<unk>'
                p = 0
                for w, wp in self.unigrams:
                    p = p + wp
                    if p > r:
                        altword = w
                        break
                nearlist.append(altword)
            #nearlist = None

            # check if there are any neighbors at all
            if nearlist == None:
                continue

            # add each neighbor to the lattice
            for widx, w in enumerate(nearlist):
                if w in self.lmfst.input_symbols() and w != word:
                    w_id = self.lmfst.input_symbols().find(w)
                    arc = fst.Arc(w_id, w_id, 0, self.get_state_id(idx+1, altfst))
                    altfst.add_arc(self.get_state_id(idx, altfst), arc)

        # mark the final state in the FST
        altfst.set_final(len(words))
        altfst.set_start(0)

        # sort lattice prior to rescoring
        altfst.arcsort()

        # rescore the lattice using the language model
        scoredfst = fst.compose(self.lmfst, altfst)

        # get best paths in the rescored lattice
        bestpaths = fst.shortestpath(scoredfst, nshortest=numalts)
        bestpaths.rmepsilon()

        altstrings = {}

        # get the strings and weights from the best paths
        for i, path in enumerate(self.paths(bestpaths)):
            path_string = ' '.join((bestpaths.input_symbols().find(arc.ilabel)).decode('utf-8') for arc in path)
            path_weight = functools.reduce(operator.add, (float(arc.weight) for arc in path))
            if not path_string in altstrings:
                altstrings[path_string] = path_weight
        
        # sort strings by weight
        scoredstrings = []
        for sent in altstrings:
            score = altstrings[sent]
            scoredstrings.append((score, sent))
        scoredstrings.sort()
        
        if len(scoredstrings) > numalts:
            scoredstrings = scoredstring[:numalts]
        
        return scoredstrings

    # helper function to check if state is in FST and add state if not
    def get_state_id(self, state, f):
        if state in f.states():
            return state
        s = f.add_state()
        return s

    # helper function to conduct depth first search on all paths in an FST
    def get_paths(self, state, f, prefix=()):
        if float(f.final(state)) != float('inf'):
            yield prefix
        for arc in f.arcs(state):
            for path in self.get_paths(arc.nextstate, f, prefix+(arc,)):
                yield path

    # get list of all paths in FST f
    def paths(self, f):
        return self.get_paths(f.start(), f)

def main(argv):
    fstfname = ''
    fname = ''

    try:
        opts, args = getopt.getopt(argv, "hu:f:")
    except getopt.GetoptError:
        print("awer.py -u <unigram_probabilities> -f <language_model_fst>")
        sys.exit(1)
    for opt, arg in opts:
        if opt == '-h':
            print("awer.py -u <unigram_probabilities> -f <language_model_fst>")
            sys.exit()
        elif opt == '-u':
            fname = arg
        elif opt == '-f':
            fstfname = arg

    if fname == '' or fstfname == '':
        print("awer.py -u <word_vectors_txt> -f <language_model_fst>")
        sys.exit(1)

    lv = AlterSent(fname, fstfname, 50000)
    #print("Ready")
    totalerr = 0
    linecnt = 0
    for line in sys.stdin:
        linecnt += 1
        words = tokenizer.word_tokenize(line)
        lines = lv.fst_alter_sent(words,1)
        toks = lines[0][1].split()
        err = 0
        for i in range(len(words)):
            if words[i] != toks[i]:
                err += 1
            
        if len(words) > 0:
            totalerr += err / len(words)
    if linecnt > 0:
        totalerr = totalerr / linecnt
    print("AWER: %.5f" % totalerr)

if __name__ == "__main__":
    main(sys.argv[1:])
