# wordvecutil: Basic operations for word vectors
#  - loading, cosine similarity, nearest neighbors

import numpy
import sys
import getopt

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
        with open(fname) as f:
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
        near = sorted([(dist[i], self.idx2word[i]) for i in range(len(dist))], reverse=True)

        # trim results and return
        if numnear > len(near):
            numnear = len(near)
            
        return near[0:numnear]

    # sim returns the cosine similarity between two words.
    # because our vectors are normalized, we can just
    # use the dot product and we are done
    
    def sim(self, w1, w2):
        if not w1 in self.word2idx:
            return None
        if not w2 in self.word2idx:
            return None
        return self.v[self.word2idx[w1]].dot(self.v[self.word2idx[w2]])

# a sample driver

def main(argv):
    fname = ''
    try:
        opts, args = getopt.getopt(argv, "hv:")
    except getopt.GetoptError:
        print("word_vectors.py -v <word_vectors_txt>")
        sys.exit(1)
    for opt, arg in opts:
        if opt == '-h':
            print("word_vectors.py -v <word_vectors_txt>")
            sys.exit()
        elif opt == '-v':
            fname = arg

    if fname == '':
        print("word_vectors.py -v <word_vectors_txt>")
        sys.exit()

    print("Loading...")

    # create the vectors from a file in text format,
    # and load at most 100000 vectors
    v = word_vectors(fname, 100000)
    print("Done.")

    # find the 10 nearest neighbors for "computer"
    print(v.nearest('computer', 10))

if __name__ == "__main__":
    main(sys.argv[1:])

