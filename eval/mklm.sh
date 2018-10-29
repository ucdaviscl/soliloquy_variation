./singletonunk.py 1 < $1 > tmp.unk
ngramsymbols < tmp.unk > tmp.syms
./ngram.sh --smooth_method=witten_bell --itype=text_sents --otype=lm --order=3 --ifile=tmp.unk --ofile=$2 --symbols=tmp.syms
