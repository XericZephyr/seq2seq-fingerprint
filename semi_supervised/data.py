"""Prepare data for seq2seq training."""

from __future__ import print_function

import os

from tensorflow.models.rnn.translate.data_utils import (
    create_vocabulary, initialize_vocabulary, data_to_token_ids)
import tensorflow as tf


tf.app.flags.DEFINE_string(
    "logp_path", "/smile/nfs/projects/nih_drug/data/logp/logp.smi", "logp data path.")
tf.app.flags.DEFINE_string(
    "pm2_path", "/smile/nfs/projects/nih_drug/data/pm2/pm2.txt", "pm2 data path")
tf.app.flags.DEFINE_string(
    "data_dir", "/tmp/seq2seq-fp/pretrain/", "Pretrain data path, holding temporary data.")

FLAGS = tf.app.flags.FLAGS

MAX_SMILE_VOCAB_TOKEN = 10000


def mkdirp(dir_path):
    """Error-free version of os.makedirs."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def logp_data_iter(logp_path=FLAGS.logp_path):
    """Yield logp SMILE representation."""
    with open(logp_path) as fobj:
        for line in fobj:
            if not len(line.strip()):
                continue
            _smile = line.strip().split()[0]
            yield _smile


def pm2_data_iter(pm2_path=FLAGS.pm2_path):
    """Yield pm2 SMILE representation."""
    with open(pm2_path) as fobj:
        for line in fobj:
            if not len(line.strip()):
                continue
            _smile = line.strip().split()[-1]
            yield _smile


def build_data_tmp(data_iter, data_path):
    """Build temp data file inside the data_directory."""
    with open(data_path, "w+") as fobj:
        for _smile in data_iter:
            fobj.write("%s\n" % _smile)


def smile_tokenizer(line):
    """Return each non-empty character as the token."""
    return list(line.strip().replace(" ", ""))


def get_vocabulary(data_path, vocab_path):
    """Get the vocabulary for specific data temp file. If not, create one."""
    # Create vocabulary if needed.
    create_vocabulary(vocab_path, data_path, MAX_SMILE_VOCAB_TOKEN,
                      tokenizer=smile_tokenizer, normalize_digits=False)
    # Return the create vocabulary.
    return initialize_vocabulary(vocab_path)


def prepare_seq2seq_pretrain_data(data_dir=FLAGS.data_dir):
    """Prepare the data iteration."""
    mkdirp(data_dir)
    # Build pm2 data set for training.
    print("Building pm2 data set...")
    pm2_data_tmp_path = os.path.join(data_dir, "pm2.tmp")
    pm2_vocab_path = os.path.join(data_dir, "pm2.vocab")
    pm2_tokens_path = os.path.join(data_dir, "pm2.tokens")
    build_data_tmp(pm2_data_iter(), pm2_data_tmp_path)
    get_vocabulary(pm2_data_tmp_path, pm2_vocab_path)
    data_to_token_ids(pm2_data_tmp_path, pm2_tokens_path, pm2_vocab_path,
                      tokenizer=smile_tokenizer, normalize_digits=False)
    # Use logp data set for development and testing.
    print("Building logp data set...")
    logp_data_tmp_path = os.path.join(data_dir, "logp.tmp")
    logp_tokens_path = os.path.join(data_dir, "logp.tokens")
    build_data_tmp(logp_data_iter(), logp_data_tmp_path)
    data_to_token_ids(logp_data_tmp_path, logp_tokens_path, pm2_vocab_path,
                      tokenizer=smile_tokenizer, normalize_digits=False)


def main(_):
    """Main function for this file."""
    prepare_seq2seq_pretrain_data()


if __name__ == "__main__":
    tf.app.run()
