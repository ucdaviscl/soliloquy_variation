#!/usr/bin/env python3

import sys
import wordvecutil
import tokenizer
import fst
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
        self.lmfst = fst.read_std(lmfname)
        self.maxtypes = maxtypes

    def fst_alter_sent(self, words, numalts=5):
        # with NLTK we could do POS tagging here
        # pos = nltk.pos_tag(text)

        # instead, we just make everything NN
        pos = [(w, 'NN') for w in words]

        altfst = fst.Acceptor(syms=self.lmfst.isyms)
        
        for idx, (word, tag) in enumerate(pos):
            if ( tag.startswith('NN') or \
                 tag.startswith('JJ') or tag.startswith('RB') or \
                 tag.startswith('VB') ) and \
                word not in ['have', 'has', 'had', 'is', 'are', 'am', \
                             'was', 'were', 'be', '.', ',', ':', '?', \
                             '!', '-', '--', 'of'] and \
                not word.startswith("'"):
                nearlist = self.vecs.near(word, 5)
                for widx, (dist, w) in enumerate(nearlist):
                    if dist > 0.1: # and w != word:
                        altfst.add_arc(idx, idx+1, w, math.log(dist) * -1)
            else:
                altfst.add_arc(idx, idx+1, word, 0)
        altfst[len(words)].final = True
        scoredfst = self.lmfst.compose(altfst)
        bestpaths = scoredfst.shortest_path(numalts)
        bestpaths.remove_epsilon()

        altstrings = {}

        for i, path in enumerate(bestpaths.paths()):
            path_string = ' '.join(bestpaths.isyms.find(arc.ilabel) for arc in path)
            path_weight = functools.reduce(operator.mul, (arc.weight for arc in path))
            if not path_string in altstrings:
                altstrings[path_string] = path_weight
            #elif altstrings[path_string] < path_weight:
            #    altstrings[path_string] = path_weight

            if len(altstrings) == numalts:
                break

        scoredstrings = []
        for str in altstrings:
            score = float(("%s" % altstrings[str]).split('(')[1].strip(')'))
            scoredstrings.append((score, str))
        scoredstrings.sort()
        
        return scoredstrings

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
            print(i, ':', score, ':', str)

        print()

if __name__ == "__main__":
    main(sys.argv[1:])
