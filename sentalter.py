#!/usr/bin/env python3

import sys
import wordvecutil
import tokenizer
import fst
import math
import functools
import operator
import getopt

import subprocess
import os
import argparse

import kenlm

# we can optionally use nltk to tag the text
# and focus on replacement of specific categories
#
# import nltk

parser = argparse.ArgumentParser(description='Sentence variation')
parser.add_argument('-v', '--vectors', type = str, default = '', help = 'word vectors', required = True)
parser.add_argument('-f', '--fst_lm', type = str, default = '', help = 'fst language model', required = True)
parser.add_argument('-d', '--onmt_dir', type = str, default = '', help = 'OpenNMT intallation directory')
parser.add_argument('-m', '--onmt_lm', type = str, default = '', help = 'OpenNMT language model')
parser.add_argument('-k', '--kenlm', type = str, default = '', help = 'KenLM model')

class AlterSent:
    def __init__(self, vecfname, lmfname, onmt_dir, model_dir, kenlm_loc, maxtypes=0):
        self.vecs = wordvecutil.word_vectors(vecfname, maxtypes)
        self.lmfst = fst.read_std(lmfname)
        self.maxtypes = maxtypes
        # self.onmt_dir = '/data/OpenNMT'
        # self.onmt_model = '/data/soliloquy_variation/language_model/luamodel_1/model_epoch13_1.16.t7'
        self.onmt_dir = onmt_dir
        self.onmt_model =  model_dir
        self.kenlm_loc = kenlm_loc
        if self.onmt_model != '':
            self.sent_rescore = self.sent_rescore_onmt
            self.onmt_model = os.path.abspath(self.onmt_model)
        elif self.kenlm_loc != '':
            self.sent_rescore = self.sent_rescore_kenlm
        else:
            self.sent_rescore =	self.sent_rescore_dummy


    def sent_rescore_onmt(self, sents):
        currdir = os.getcwd()
        os.chdir(self.onmt_dir)
        with open('sentalter_input.txt', 'w') as fout:
            fout.write('\n'.join([i[1] for i in sents]))
        subprocess.call(['th', 'lm.lua', '-log_level', 'ERROR', '-gpuid', '1', '-model', self.onmt_model, '-src', 'sentalter_input.txt', '-output', 'sentalter_output.txt', 'score'])
        with open('sentalter_output.txt') as fin:
            scores = []
            for line in fin:
                scores.append(float(line.rstrip('\n')))
        nscoredsent = [[scores[i], sents[i][0], sents[i][1]] for i in range(len(sents))]
        os.chdir(currdir)
        return nscoredsent

    def sent_rescore_kenlm(self, sents):
        model = kenlm.Model(self.kenlm_loc)
        nscoredsent = [[model.perplexity(sents[i][1]), sents[i][0], sents[i][1]] for i in range(len(sents))]
        return nscoredsent

    def sent_rescore_dummy(self, sents):
        nscoredsent = [[sents[i][0], sents[i][0], sents[i][1]] for i in range(len(sents))]
        return nscoredsent

    def fst_alter_sent(self, words, numalts=5, cutoff = 0):
        # with NLTK we could do POS tagging here
        # pos = nltk.pos_tag(text)

        # instead, we just make everything NN
        pos = [(w, 'NN') for w in words]

        altfst = fst.Acceptor(syms=self.lmfst.isyms)
        
        for idx, (word, tag) in enumerate(pos):
            # add the word to the lattice
            if word in altfst.isyms:
                altfst.add_arc(idx, idx+1, word, 0)
            else:
                altfst.add_arc(idx, idx+1, "<unk>", 0)

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
                    if dist > 0.1 and w in altfst.isyms and w != word:
                        altfst.add_arc(idx, idx+1, w, (math.log(dist) * -1)/1000)

        # mark the final state in the FST
        altfst[len(words)].final = True

        # rescore the lattice using the language model
        scoredfst = self.lmfst.compose(altfst)

        # get best paths in the rescored lattice
        bestpaths = scoredfst.shortest_path(numalts)
        bestpaths.remove_epsilon()

        altstrings = {}

        # get the strings and weights from the best paths
        for i, path in enumerate(bestpaths.paths()):
            path_string = ' '.join(bestpaths.isyms.find(arc.ilabel) for arc in path)
            path_weight = functools.reduce(operator.mul, (arc.weight for arc in path))
            if not path_string in altstrings:
                altstrings[path_string] = path_weight

        # print('Altstrings:')
        # print(altstrings)

        # sort strings by weight
        scoredstrings = []
        for sent in altstrings:
            score = float(("%s" % altstrings[sent]).split('(')[1].strip(')'))
            scoredstrings.append((score, sent))

        scoredstrings = self.sent_rescore(scoredstrings)
        scoredstrings.sort()

        if len(scoredstrings) > numalts:
            scoredstrings = scoredstring[:numalts]

        if cutoff > 0:
            scoredstrings = [s for s in scoredstrings if s[0] <= cutoff]
        
        # print('Scoredstrings:')
        # print(scoredstrings)
        return scoredstrings

def main():
    params = parser.parse_args()

    print('Processing...')
    lv = AlterSent(params.vectors, params.fst_lm, params.onmt_dir, params.onmt_lm, params.kenlm, 50000)
    print("Ready")
    try:
        while True:
            line = input()
            if line.rstrip(' \n') == '':
                continue
            print()
            words = tokenizer.word_tokenize(line)
            lines = lv.fst_alter_sent(words,100)

            for i, (newscore, score, sent) in enumerate(lines):
                print(i, ':', '%.3f' % newscore, ':', '%.3f' % score, ':', sent)

            print()
    except EOFError:
        pass

if __name__ == "__main__":
    main()
