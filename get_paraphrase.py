import os
import sys

import tensorflow as tf
import opennmt  as onmt
from opennmt.runner import Runner
from opennmt.config import load_model, load_config

import tokenizer
from paraphrase.train_model import NMTCustom, load_custom_model

tf.logging.set_verbosity('INFO')

os.chdir('paraphrase')
model_dir = 'model_2/'

config = load_config(['model_2.yml'])
model = load_custom_model(model_dir, NMTCustom())
session_config = tf.ConfigProto(intra_op_parallelism_threads=0, inter_op_parallelism_threads=0)
runner = Runner(model, config, seed = None, num_devices = 1, gpu_allow_growth = False, session_config = session_config)

# try:
# 	while True:
# 		line = input()
# 		if line.rstrip(' \n') == '':
# 			continue
# 		print()
# 		words = tokenizer.word_tokenize(line)
# 		with open('tmp.txt', 'w') as tempout:
# 			tempout.write(' '.join(words)+'\n')
# 		runner.infer('tmp.txt')
# 		os.remove('tmp.txt')

# 		print()
# except EOFError:
# 	pass

rawtrain = 'dstc2_user_train_{}.txt'
paraout = 'dstc2_user_train_nmtparaphrase_{}.txt'
for i in range(1,101):
	runner.infer(os.path.join('dstc2_tests', rawtrain.format(i)), predictions_file = os.path.join('dstc2_tests', paraout.format(i)))