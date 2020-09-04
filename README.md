# Soliloquy Variation

- Generation of lexical and sentence-level variety using word vectors, sequence to sequence models, and transducers.
- Tools for validating generated sentence variations.

## Dependencies

- Python >= 3.6
- transformers
- lm-score

## External data

Models and data are available here (password required):
https://ucdavis.box.com/v/soliloquy 

Get en_vec.txt, en4M.v100k.fst and en4M.v100k.vocab.

## Lexical variation

Usage:

```
$ python3 lexalter.py -v [word embeddings file]
```

A word embeddings file `en_vec.txt` is available in the data folder. The file is expected to be in the word2vec text format. We assume the words are ordered from most to least frequent. Input from stdin is a word followed by several words that provide the context. For example, the input:
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

## Sentence Variation

### Using FSTs

To get sentence variation using word vectors and a language model, use sentalter.py. It includes a sample driver, invoked as follows:

```
python sentalter.py -v en_vec.txt -f en4M.v100k.fst -s en4M.v100k.vocab
```

where en_vec.txt contains word vectors in text format, and en4M.v100k.fst is a 4-gram language model encoded as an FST. Input for the sample driver is one sentence per line in stdin, and the output is a scored list of alternative sentences.



## Testing (out of date)

### Testing the effects of reducing size of training file

We use [KenLM](https://github.com/kpu/kenlm) for fast language model evaluation, and you need to install both C++ binaries and the Python module. [`conduct_experiment.py`](paraphrase/conduct_experiment.py) tests the effects of data size by training language models using 1%, 2%, ..., 100% of the data and testing those models with a development set (number of partitions and how large each part is can be set). To generate KenLM models:

```
$ python3 conduct_experiment.py train -b [KenLM binary location] -s [souce] -t --gen_dev [development file] -f [folder to store generated files]
```

KenLM binary location is where compiled programs (e.g. `lmplz` and `query`) are stored. After models are generated, to test them:

```
$ python3 conduct_experiment.py evaluate -f [folder] -d [development set] -r [results file]
```
