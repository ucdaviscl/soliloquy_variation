#!/usr/bin/env python3

import sys
import tokenizer
import wordvecutil
import math
import getopt

# AlterLex contains at most maxtypes word vectors.
# the word vector file is assumed to be in word2vec format,
# with vectors appearing from most to least frequent
# (which is standard in word2vec and fasttext)
#
# fname: the name of the word vector file in text format
# maxtypes: maximum number of vectors to load

class AlterLex:
    def __init__(self, fname, maxtypes=0):
        self.vecs = wordvecutil.word_vectors(fname, maxtypes)
        self.maxtypes = maxtypes

    # alter returns a list of alternative words to the main word
    #
    # mainword: the main word
    # words: a list of context words
    
    def alter(self, mainword, words, distflag=False, numalts=10):

        # find a list of nearest neighbords to the main word
        nearlist = self.vecs.near(mainword, 100)

        # was the main word found?
        if nearlist == None:
            return None

        # go through the context words
        for widx, word in enumerate(words):
            # adjust the score of each neighbor according to
            # similarity to the current context word
            for idx, (d, w) in enumerate(nearlist):

                # distance to context word
                dist = self.vecs.sim(word, w)

                # was the context word found?
                if dist == None:
                    continue

                # should the influence of the context word
                # get weaker as we go down the list?
                if distflag:
                    nearlist[idx] = (d * math.pow(dist, 1/(widx+2)), w)
                else:
                    nearlist[idx] = (d * dist, w)

        # now sort the neighbors according to final score
        nearlist.sort(key=lambda x: x[0], reverse=True)
        if numalts > len(nearlist):
            numalts = len(nearlist)
            
        return nearlist[0:numalts]

# a sample driver

def main(argv):
    fname = ''

    try:
        opts, args = getopt.getopt(argv, "hv:")
    except getopt.GetoptError:
        print("lexalter.py -v <word_vectors_txt>")
        sys.exit(1)
    for opt, arg in opts:
        if opt == '-h':
            print("lexalter.py -v <word_vectors_txt>")
            sys.exit()
        elif opt == '-v':
            fname = arg

    if fname == '':
        print("lexalter.py -v <word_vectors_txt>")
        sys.exit(1)

    lv = AlterLex(fname, 50000)

    # get a main word and some context words from stdin
    for line in sys.stdin:
        print()

        words = tokenizer.word_tokenize(line)
        mainword = words.pop(0)

        # get the alternatives
        nearlist = lv.alter(mainword, words, 10)

        # print the alternatives
        if nearlist != None:
            for (idx, w) in enumerate(nearlist):        
                print(w[1])

        print()
        print('--------------')
        print()
            
if __name__ == "__main__":
    main(sys.argv[1:])
