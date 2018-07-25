Soliloquy Variation Quickstart
==========

Prerequisites
----------
 * Python 3.x
 * gcc, g++ >= 4.1

Installing Dependencies
----------
1. Install [numpy](http://www.numpy.org/)
 ```
    pip3 install numpy
 ```

2. Download [OpenFST](http://www.openfst.org) 1.3.4 (later versions are not compatible)
 ```
    wget http://openfst.org/twiki/pub/FST/FstDownload/openfst-1.3.4.tar.gz
 ```
    
3. Extract OpenFST and enter its directory
 ```
    tar xzf openfst-1.3.4.tar.gz
    cd openfst-1.3.4
 ```
    
4. Configure, compile, and install OpenFST.
 ```
    ./configure
    make
    make install
 ```
	
5. Install [pyfst](https://pyfst.github.io/)
 ```
    pip3 install pyfst
 ```

Use
----------
Lexical and sentence alternatives can be computed with ```lexalter.py``` and ```sentalter.py```, both of which require additional models.

### Models
Required models are available [here](https://ucdavis.app.box.com/v/soliloquy/folder/48246129089) (password required). Both ```lexalter.py``` and ```sentalter.py``` require ```en_vec.txt```, a text file containing word vectors, ordered from most to least frequent, in word2vec format. ```sentalter.py``` further requires ```wiki_other_en.o4.h10m.fst```, a 4-gram language model encoded as an OpenFST finite-state transducer.

### lexalter.py
```lexalter.py``` finds alternatives to words, optionally taking into account additional context words, and can be run as:
 ```
    python3 lexalter.py -v en_vec.txt
 ```
Input via stdin a word, optionally followed by one or more space-separated context words. A newline-separated list of lexical alternatives, with the best alternative listed first, will be output.

### sentalter.py
```sentalter.py``` finds alternatives to given sentences, and can be run as:
 ```
    python3 sentalter.py -v en_vec.txt -f wiki_other_en.o4.h10m.fst
 ```
Once the model has been loaded, indicated in stdout by "Ready," input a sentence. A newline-separated list of ranked and scored alternative sentences will be output.
