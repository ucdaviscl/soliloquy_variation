# Soliloquy Variation
Generation of lexical and sentence-level variety using word vectors, sequence to sequence models, and transducers

## External data
Models and data are available here (password required):
https://ucdavis.box.com/v/soliloquy (to-do: add OpenNMT language model)

## Lexical variation
To get alternatives to a word, use lexalter.py. A file containing word vectors in text format is required (en_vec.txt in the data URL above). The program includes a sample driver that can be invoked like this:
```
python lexalter.py -v en_vec.txt
```

Here, en_vec.txt is a file containing word vectors using the word2vec text format. We assume the words are ordered from most to least frequent. Input from stdin is a word followed by several words that provide the context. For example, the input:
```
orange
```
produces the following output:
```
orange
yellow
purple
blue
brown
green
pink
pale
yellowish
reddish
```
and the input:
```
orange apple banana
```
produces:
```
orange
peach
lemon
oranges
strawberry
ginger
cherry
cream
green
olive
```

## Sentence variation
### Using FSTs
To get sentence variation using word vectors and a language model, use sentalter.py. It includes a sample driver, invoked as follows:
```
python sentalter.py -v en_vec.txt -f wiki_other_en.o4.h10m.fst
```
where en_vec.txt contains word vectors in text format, and wiki_other_en.o4.h10m.fst is a 4-gram language model encoded as an OpenFST finite-state transducer. Input for the sample driver is one sentence per line in stdin, and the output is a scored list of alternative sentences.

#### Requirements
All programs require python3.

sentalter.py requires [PyFST](https://pyfst.github.io), which requires [OpenFST](https://openfst.org) version 1.3 [(direct link to download)](http://openfst.org/twiki/pub/FST/FstDownload/openfst-1.3.4.tar.gz). Newer versions of OpenFST do not work with PyFST.

However, a fork of PyFST is [available](https://github.com/placebokkk/pyfst) and supports the latest version of OpenFST.

Using the neural language model requires [OpenNMT](https://github.com/OpenNMT/OpenNMT/), see the installation instructions. Note that GPU support is required to use the pretrained model.

sentalter.py has only been tested in ubuntu linux, and lexalter.py has been tested in ubuntu linux and mac os.

### Using sequence to sequence models
The external data directory above contains neural models for text simplification that work with [OpenNMT](http://opennmt.net). The README.txt file in the seq2seq subdirectory in the data URL contains instructions for using the sequence to sequence models.
