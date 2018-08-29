import sys
import argparse
import pickle

import sentalter
import tokenizer

print('Initializing...')
# lv = sentalter.AlterSent('en_vec.txt', 'wiki_other_en.o4.h10m.fst', '/data/OpenNMT', '/data/soliloquy_variation/language_model/luamodel_1/model_epoch13_1.16.t7', 50000)
lv = sentalter.AlterSent('en_vec.txt', 'wiki_other_en.o4.h10m.fst', '', '', 'language_model/kenlm_model/modelo5.binary', 50000)

eprint = lambda x: print(x, file = sys.stderr)

trainfn = '/data/soliloquy_variation/paraphrase/dstc2_tests/dstc2_user_train_{}.txt'
outfmt = '/data/soliloquy_variation/paraphrase/dstc2_tests/dstc2_user_train_fstparaphrase_tenbest_{}.txt'
source = '/data/soliloquy_variation/paraphrase/dstc6/dstc6_user_train_5.txt'
conservative = True
n_best = 10

def gen_all_paraphrases():
    with open(source) as fin:
        sents = dict()
        for line in fin:
            l = line.rstrip('\n ')
            if l not in sents:
                sents[l]  = dict()
                sents[l]['num'] = 0
            sents[l]['num']+=1
    if conservative:
        baseline = lv.sent_rescore([['', x] for x in sents])
        for s in baseline:
            sents[s[2]]['baseline'] = s[0]
        print('baseline complete')

    for i, sent in enumerate(sents):
        sys.stdout.write('\rParaphrasing {}/{}'.format(i, len(sents)))
        words = tokenizer.word_tokenize(sent)
        
        if conservative:
            lines = lv.fst_alter_sent(words, n_best, cutoff = sents[sent]['baseline'])
        else:
            lines = lv.fst_alter_sent(words, n_best)
        
        sents[sent]['para'] = lines

    print()
    with open('dstc6_100_parafst.pickle', 'wb') as pout:
        pickle.dump(sents, file = pout)

    with open('dstc6_100_parafst.txt', 'w') as fout:
        for line in sents:
            for x in sents[line.rstrip('\n ')]['para']:
                fout.write(x[2] + '\n')

def get_training_files():
    with open('dstc2_parafst.pickle', 'rb') as pin:
        sents = pickle.load(pin)

    for i in range(1,101):
        sys.stdout.write('\rGenerating training data {}/100'.format(i))
        with open(outfmt.format(i), 'w') as fout:
            with open(trainfn.format(i)) as fin:
                for sent in fin:
                    for x in sents[sent.rstrip('\n ')]['para']:
                        fout.write(x[2] + '\n')
    print()

if __name__ == '__main__':
    gen_all_paraphrases()
    # get_training_files()
