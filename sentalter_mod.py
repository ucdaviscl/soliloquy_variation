#!/usr/bin/env python3

import sys
import wordvecutil
import tokenizer
import pywrapfst as fst
import math
import functools
import operator
import getopt

# we can optionally use nltk to tag the text
# and focus on replacement of specific categories
#
# import nltk

class AlterSent:
    def __init__(self, vecfname, lmfname, maxtypes=0):
        self.vecs = wordvecutil.word_vectors(vecfname, maxtypes)
        self.lmfst = fst.Fst.read(lmfname)
        self.maxtypes = maxtypes

    def fst_alter_sent(self, words, numalts=5):
        # with NLTK we could do POS tagging here
        # pos = nltk.pos_tag(text)

        # instead, we just make everything NN
        pos = [(w, 'NN') for w in words]

        altfst = fst.Fst()
        altfst.add_state()
        altfst.set_input_symbols(self.lmfst.input_symbols())
        altfst.set_output_symbols(self.lmfst.output_symbols())

        syms = altfst.input_symbols()
        
        for idx, (word, tag) in enumerate(pos):
            # add the word to the lattice
            if word in syms:
                word_id = syms.find(word)
                arc = fst.Arc(word_id, word_id, 0, self.get_state_id(idx+1, altfst))
                altfst.add_arc(self.get_state_id(idx, altfst), arc)
            else:
                word_id = syms.find("<unk>")
                arc = fst.Arc(word_id, word_id, 0, self.get_state_id(idx+1, altfst))
                altfst.add_arc(self.get_state_id(idx, altfst), arc)

            # add word alternatives to the lattice
            if ( tag.startswith('NN') or \
                 tag.startswith('JJ') or tag.startswith('RB') or \
                 tag.startswith('VB') ) and \
                word not in ['have', 'has', 'had', 'is', 'are', 'am', \
                             'was', 'were', 'be', '.', ',', ':', '?', \
                             '!', '-', '--', 'of'] and \
                not word.startswith("'"):
                nearlist = self.vecs.near(word, 5)

                # check if there are any neighbors at all
                if nearlist == None:
                    continue

                # add each neighbor to the lattice
                for widx, (dist, w) in enumerate(nearlist):
                    if dist > 0.1 and w in altfst.input_symbols() and w != word:
                        w_id = syms.find(w)
                        arc = fst.Arc(w_id, w_id, (math.log(dist) * -1)/1000, self.get_state_id(idx+1, altfst))
                        altfst.add_arc(self.get_state_id(idx, altfst), arc)

        # mark the final state in the FST
        altfst.set_final(len(words))
        altfst.set_start(0)
        
#        print(self.lmfst.num_states())
#        print(altfst.num_states())

        self.lmfst.arcsort()
        altfst.arcsort()

#        for state in altfst.states():
#            for arc in altfst.arcs(state):
#                print(state, syms.find(arc.ilabel), syms.find(arc.olabel), arc.nextstate, float(arc.weight))

        # rescore the lattice using the language model
        scoredfst = fst.compose(self.lmfst, altfst)
#        print(scoredfst.num_states())

        # get best paths in the rescored lattice
#        bestpaths = fst.Fst()
#        bestpaths.set_input_symbols(self.lmfst.input_symbols())
#        bestpaths.set_output_symbols(self.lmfst.output_symbols())
        bestpaths = fst.shortestpath(scoredfst, numalts)
#        bestpaths.rmepsilon()
        bestpaths.set_final(0)

        print(bestpaths.num_states())
        print(bestpaths.start())

#        for state in bestpaths.states():
#            print(state)
#            for arc in bestpaths.arcs(state):
#                print(arc.ilabel, arc.olabel, arc.nextstate, float(arc.weight))

        altstrings = {}

        # get the strings and weights from the best paths
        for i, path in enumerate(self.paths(bestpaths)):
            print(path)
            path_string = ' '.join(bestpaths.input_symbols().find(arc.ilabel) for arc in path)
            path_weight = functools.reduce(operator.mul, (arc.weight() for arc in path))
            if not path_string in altstrings:
                altstrings[path_string] = path_weight

        # sort strings by weight
        scoredstrings = []
        for str in altstrings:
            score = float(("%s" % altstrings[str]).split('(')[1].strip(')'))
            scoredstrings.append((score, str))
        scoredstrings.sort()
        
        if len(scoredstrings) > numalts:
            scoredstrings = scoredstring[:numalts]
        
        return scoredstrings

    def get_state_id(self, state, f):
        if state in f.states():
            return state
        s = f.add_state()
        return s

    def get_paths(self, state, f, prefix=()):
        print("get_paths")
        print("states ", f.num_states())
        print("state ", state)
        print("final", float(f.final(state)))
        print("arcs ", f.num_arcs(state))
        if float(f.final(state)) == 0:
            print("get_paths_end")
            yield prefix
        for arc in f.arcs(state):
            for path in self.get_paths(arc.nextstate, f, prefix+(arc,)):
                print("recursed")
                print(path)
                yield path

    def paths(self, f):
        print("paths")
        return self.get_paths(f.start(), f)

def main(argv):
    fstfname = ''
    fname = ''

    try:
        opts, args = getopt.getopt(argv, "hv:f:")
    except getopt.GetoptError:
        print("lexalter.py -v <word_vectors_txt> -f <language_model_fst>")
        sys.exit(1)
    for opt, arg in opts:
        if opt == '-h':
            print("lexalter.py -v <word_vectors_txt> -f <language_model_fst>")
            sys.exit()
        elif opt == '-v':
            fname = arg
        elif opt == '-f':
            fstfname = arg

    if fname == '' or fstfname == '':
        print("lexalter.py -v <word_vectors_txt> -f <language_model_fst>")
        sys.exit(1)

    lv = AlterSent(fname, fstfname, 50000)
    print("Ready")
    for line in sys.stdin:
        print()
        words = tokenizer.word_tokenize(line)
        lines = lv.fst_alter_sent(words,100)

        for i, (score, str) in enumerate(lines):
            print(i, ':', '%.3f' % score, ':', str)

        print()

if __name__ == "__main__":
    main(sys.argv[1:])
