./singletonunk.py 1 < $1 > tmp/tmp.unk
ngramsymbols < tmp/tmp.unk > tmp/tmp.syms
./ngram.sh --smooth_method=witten_bell --itype=text_sents --otype=lm --order=3 --ifile=tmp/tmp.unk --ofile=$2 --symbols=tmp/tmp.syms
