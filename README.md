# Soliloquy Variation

- Generation of lexical and sentence-level variety using word vectors, sequence to sequence models, and transducers.
- Tools for validating generated sentence variations.

## Dependencies

- Python >= 3.6
- [PyFST](https://github.com/placebokkk/pyfst) with OpenFST (required for fst sentence variation)
- [OpenNMT](https://github.com/OpenNMT/OpenNMT) with GPU support (required for neural language model)
- [KenLM](https://github.com/kpu/kenlm) and its Python module (required for paraphrase testing)
- [OpenNMT-tf](https://github.com/OpenNMT/OpenNMT-tf) + [Tensorflow](https://www.tensorflow.org/) (required for neural machine paraphrasing)

## External data

Models and data are available here (password required):
https://ucdavis.box.com/v/soliloquy 

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
python sentalter.py -v en_vec.txt -f wiki_other_en.o4.h10m.fst
```

where en_vec.txt contains word vectors in text format, and wiki_other_en.o4.h10m.fst is a 4-gram language model encoded as an OpenFST finite-state transducer. Input for the sample driver is one sentence per line in stdin, and the output is a scored list of alternative sentences.

#### Generating batch of paraphrases

Use [`get_fst_paraphrases.py`](get_fst_paraphrases.py) to generate paraphrases sentence by sentence from an input file. You can use either an OpenNMT or a KenLM language model to rescore generated sentences.

```
$ python3 get_fst_paraphrase.py -v [word vectors] -f [fst model] -i [input] -o [output]
```

### Using sequence to sequence models

The external data directory above contains neural models for text simplification that work with [OpenNMT](http://opennmt.net). The README.txt file in the seq2seq subdirectory in the data URL contains instructions for using the sequence to sequence models.

### Using neural machine paraphrase models

#### Training and testing

1. To train a neural machine paraphrase model, you will need a training set containing paraphrases. We used the 5.3M+ processed and filtered data by [John Wieting](https://www.cs.cmu.edu/~jwieting/). You can use [`split_input.py`](paraphrase/split_input.py) to split the data into corresponding pairs 

2. Generate vocabularies for training and development sets using [`train_model.py`](paraphrase/train_model.py)

   ```
   $ python3 train_model.py get_vocab -r [source] -t [target]
   ```

3. Define the parameters of your model in a YAML file, see [`nmp_model.yml`](paraphrase/nmp_model.yml) as an example. You need to specify the save folder and locations of training files, training vocabularies, and development files (if using them).

4. Train the model.

   ```
   $ python3 train_model.py train -c [configuration file] -s [save folder]
   ```

5. Evaluate the model.

   ```
   $ python3 train_model.py evaluate -c [configuration file] -s [save folder]
   ```

#### Generating paraphrases

Use [`get_nmp_paraphrases.py`](get_nmp_paraphrases.py) to generate paraphrases sentence by sentence from an input file.

```
$ python3 get_nmp_paraphrase.py -d [model save folder] -c [configuration file] -i [input] -o [output]
```

## Testing

### Testing the effects of reducing size of training file

We use [KenLM](https://github.com/kpu/kenlm) for fast language model evaluation, and you need to install both C++ binaries and the Python module. [`conduct_experiment.py`](paraphrase/conduct_experiment.py) tests the effects of data size by training language models using 1%, 2%, ..., 100% of the data and testing those models with a development set (number of partitions and how large each part is can be set). To generate KenLM models:

```
$ python3 conduct_experiment.py train -b [KenLM binary location] -s [souce] -t --gen_dev [development file] -f [folder to store generated files]
```

KenLM binary location is where compiled programs (e.g. `lmplz` and `query`) are stored. After models are generated, to test them:

```
$ python3 conduct_experiment.py evaluate -f [folder] -d [development set] -r [results file]
```
