import argparse

import tensorflow as tf
import opennmt as onmt
from opennmt.runner import Runner
from opennmt.config import load_model, load_config

model_dir = 'model_1/'

def get_vocab():
	tokenizer = onmt.tokenizers.SpaceTokenizer()

	special_tokens = [onmt.constants.PADDING_TOKEN, onmt.constants.START_OF_SENTENCE_TOKEN, onmt.constants.END_OF_SENTENCE_TOKEN]

	invocab = onmt.utils.Vocab(special_tokens = special_tokens)
	outvocab = onmt.utils.Vocab(special_tokens = special_tokens)

	invocab.add_from_text('para5m_train.1', tokenizer=tokenizer)
	outvocab.add_from_text('para5m_train.2', tokenizer=tokenizer)

	invocab = invocab.prune(max_size=80000, min_frequency=2)
	outvocab = outvocab.prune(max_size=80000, min_frequency=2)

	invocab.serialize('para5m_train.1.vocab')
	outvocab.serialize('para5m_train.2.vocab')

def run_model():
	tf.logging.set_verbosity('INFO')

	config = load_config(['model_1.yml'])
	model = load_model(model_dir, '', model_name = 'NMTSmall')
	session_config = tf.ConfigProto(intra_op_parallelism_threads=0, inter_op_parallelism_threads=0)
	runner = Runner(model, config, seed = None, num_devices = 1, gpu_allow_growth = False, session_config = session_config)
	runner.train_and_evaluate()

run_model()
