import os

import tensorflow as tf
import opennmt  as onmt
from opennmt.runner import Runner
from opennmt.config import load_model, load_config

from .. import tokenizer

tf.logging.set_verbosity('INFO')

config = load_config(['model_1.yml'])
model = load_model(model_dir, '', model_name = 'NMTSmall')
session_config = tf.ConfigProto(intra_op_parallelism_threads=0, inter_op_parallelism_threads=0)
runner = Runner(model, config, seed = None, num_devices = 1, gpu_allow_growth = False, session_config = session_config)

try:
	while True:
		line = input()
		if line.rstrip(' \n') == '':
			continue
		print()
		words = tokenizer.word_tokenize(line)
		with open('tmp.txt', 'w') as tempout:
			tempout.write(' '.join(words)+'\n')
		runner.infer('tmp.txt')
		os.remove('tmp.txt')

		print()
except EOFError:
	pass
