import os
import pickle
import argparse

import tensorflow as tf
import opennmt as onmt
from opennmt.runner import Runner
from opennmt.config import load_model, load_config

class NMTCustom(onmt.models.SequenceToSequence):
	def __init__(self):
		super(NMTCustom, self).__init__(
			source_inputter=onmt.inputters.WordEmbedder(vocabulary_file_key="source_words_vocabulary", embedding_size=512),
			target_inputter=onmt.inputters.WordEmbedder(vocabulary_file_key="target_words_vocabulary", embedding_size=512),
			encoder=onmt.encoders.BidirectionalRNNEncoder(num_layers=4, num_units=512, reducer=onmt.layers.ConcatReducer(), cell_class=tf.contrib.rnn.LSTMCell, dropout=0.4,residual_connections=False),
			decoder=onmt.decoders.AttentionalRNNDecoder(num_layers=4, num_units=512, bridge=onmt.layers.CopyBridge(), attention_mechanism_class=tf.contrib.seq2seq.LuongAttention, cell_class=tf.contrib.rnn.LSTMCell, dropout=0.4, residual_connections=False))

def load_custom_model(model_dir, model = None):
  
  serial_model_file = os.path.join(model_dir, "model_description.pkl")

  if model:
    if tf.train.latest_checkpoint(model_dir) is not None:
      tf.logging.warn(
          "You provided a model configuration but a checkpoint already exists. "
          "The model configuration must define the same model as the one used for "
          "the initial training. However, you can change non structural values like "
          "dropout.")
    with open(serial_model_file, "wb") as serial_model:
      pickle.dump(model, serial_model)
  elif not os.path.isfile(serial_model_file):
    raise RuntimeError("A model configuration is required.")
  else:
    tf.logging.info("Loading serialized model description from %s", serial_model_file)
    with open(serial_model_file, "rb") as serial_model:
      model = pickle.load(serial_model)

  return model

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

def run_model_1():
	tf.logging.set_verbosity('INFO')
	model_dir = 'model_1/'

	config = load_config(['model_1.yml'])
	model = load_model(model_dir, '', model_name = 'NMTSmall')
	session_config = tf.ConfigProto(intra_op_parallelism_threads=0, inter_op_parallelism_threads=0)
	runner = Runner(model, config, seed = None, num_devices = 1, gpu_allow_growth = False, session_config = session_config)
	runner.train_and_evaluate()

def run_model_2(evaluate = False):
	tf.logging.set_verbosity('INFO')
	model_dir = 'model_2/'

	config = load_config(['model_2.yml'])
	model = load_custom_model(model_dir, NMTCustom())
	session_config = tf.ConfigProto(intra_op_parallelism_threads=0, inter_op_parallelism_threads=0)
	runner = Runner(model, config, seed = None, num_devices = 1, gpu_allow_growth = False, session_config = session_config)
	if evaluate:
		runner.evaluate()
	else:
		runner.train_and_evaluate()

if __name__ == '__main__':
	run_model_2(True)
