# wordvecutil: Basic operations for word vectors
#  - loading, cosine similarity, nearest neighbors, display embeddings

import numpy
import sys
import os
import argparse


class word_vectors:

    # fname: the file containing word vectors in text format
    # maxtypes: the maximum size of the vocabulary

    def __init__(self, fname, maxtypes=0):
        self.word2idx = {}
        self.idx2word = []
        self.numtypes = 0
        self.dim = 0
        self.v = self.load_vectors(fname, maxtypes)

    # load vectors from a file in text format
    # fname: the file name
    
    def load_vectors(self, fname, max=0):
        cnt = 0
        with open(fname, encoding='utf-8') as f:
            toks = f.readline().split()
            numtypes = int(toks[0])
            dim = int(toks[1])
            if max > 0 and max < numtypes:
                numtypes = max

            # initialize the vectors as a two dimensional
            # numpy array. 
            vecs = numpy.zeros((numtypes, dim), dtype=numpy.float16)

            # go through the file line by line
            for line in f:
                # get the word and the vector as a string
                word, vecstr = line.split(' ', 1)
                vecstr = vecstr.rstrip()

                # now make the vector a numpy array
                vec = numpy.fromstring(vecstr, numpy.float16, sep=' ')

                # add the normalized vector
                norm = numpy.linalg.norm(vec, ord=None)
                vecs[cnt] = vec/norm

                # index the word
                self.word2idx[word] = cnt
                self.idx2word.append(word)
            
                cnt += 1
                if cnt >= numtypes:
                    break
            
        return vecs

    # near gets the nearest neighbors of a word
    # word: the target word
    # numnear: number of nearest neighbors

    def near(self, word, numnear):

        # check if the word is in our index
        if not word in self.word2idx:
            return None

        # get the distance to all the words we know.
        dist = self.v.dot(self.v[self.word2idx[word]])

        # sort by distance
        nearest = sorted([(dist[i], self.idx2word[i]) for i in range(len(dist))], reverse=True)

        # trim results and return
        if numnear > len(nearest):
            numnear = len(nearest)
            
        return nearest[:numnear]

    # nearvec gets the nearest neighbors of a vector
    # vec: the target vector
    # numnear: number of nearest neighbors

    def nearvec(self, vec, numnear):

        # get the distance to all the words we know.
        dist = self.v.dot(vec)

        # sort by distance
        nearest = sorted([(dist[i], self.idx2word[i]) for i in range(len(dist))], reverse=True)

        # trim results and return
        if numnear > len(nearest):
            numnear = len(nearest)
            
        return nearest[:numnear]

    # sim returns the cosine similarity between two words.
    # because our vectors are normalized, we can just
    # use the dot product and we are done
    
    def sim(self, w1, w2):
        if not w1 in self.word2idx:
            return None
        if not w2 in self.word2idx:
            return None
        return self.v[self.word2idx[w1]].dot(self.v[self.word2idx[w2]])

    # return embeddings for word w
    # third argument is a placeholder for compatibility
    # consider change functions to take *args

    def getword(self, w, _):
        if not w in self.word2idx:
            print('Word {} not found'.format(w))
            return
        else:
            return w + ' ' + ' '.join([str(x) for x in self.v[self.word2idx[w]]])


# a sample driver

def main():
    parser = argparse.ArgumentParser(description='word vector utilities')
    parser.add_argument('--mode', choices = ['nearest', 'query'], default = 'nearest')
    parser.add_argument('-v' ,'--vec_file', type=str, default = '', help='word vector text')
    parser.add_argument('-f', '--input', type = str, default = '', help = 'list of queries in a file, one word in each line')
    parser.add_argument('-d', '--distance', type = int, default = 10)
    parser.add_argument('-o', '--output', type = str, default = '', help = 'output of results')
    parser.add_argument('-m', '--max', type = int, default = 0, help = 'max number of vectors to be read')
    parser.add_argument('-i', '--interactive', action = 'store_true', help = 'enable interactive mode')
    parser.add_argument('cmd_queries', nargs = '*', default = [], help = 'words to query')
    params = parser.parse_args()
    
    if not os.path.isfile(params.vec_file):
        print('Vector file \"{}\" not found\nUse -h for help'.format(params.vec_file))
        exit(1)

    print("Loading...")

    # create the vectors from a file in text format,
    v = word_vectors(params.vec_file, params.max)
    print("Done.")

    # select function based on mode
    if params.mode == 'nearest':
        qf = v.near
    else:
        qf = v.getword

    # select output function
    if len(params.output) > 0:
        f = open(params.output, 'w')
        myprint = lambda x: f.write(str(x)+'\n')
    else:
        f = None
        myprint = print


    for w in params.cmd_queries:
        myprint(qf(w, params.distance))
    
    if os.path.isfile(params.input):
        with open(params.input) as qin:
            for line in qin:
                myprint(qf(line.rstrip('\n '), params.distance))


    # if interactive mode is enabled, read input  line by line until EOF
    if params.interactive:
        try:
            while True:
                w = input()
                myprint(qf(w.rstrip('\n'), params.distance))
        except EOFError:
            pass

    if f:
        f.close()
if __name__ == "__main__":
    main()

