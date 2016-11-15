"""Prepare data for seq2seq training."""

from __future__ import print_function

import os

import tensorflow as tf


tf.app.flags.DEFINE_string(
    "logp_path", "/smile/nfs/projects/nih_drug/data/logp/logp.smi", "logp data path.")
tf.app.flags.DEFINE_string(
    "pm2_path", "/smile/nfs/projects/nih_drug/data/pm2/pm2.txt", "pm2 data path")
tf.app.flags.DEFINE_string(
    "data_dir", "/tmp/seq2seq-fp/pretrain/", "Pretrain data path, holding temporary data.")

FLAGS = tf.app.flags.FLAGS


DATA_TMP_FILE = "data.tmp"
VOCAB_FILE = "data.vocab"


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


def build_tmp_file(data_dir=FLAGS.data_dir):
    """Merge pm2 and logp SMILE representation into same file."""
    os.makedirs(data_dir)
    print("Processing temporary data file...")
    tmp_data_path = os.path.join(data_dir, DATA_TMP_FILE)
    with open(tmp_data_path, "w+") as fobj:
        print("Reading logp data...")
        logp_iter = logp_data_iter()
        for _smile in logp_iter:
            fobj.write("%s\n" % _smile.strip())
        print("Reading pm2 data...")
        pm2_iter = pm2_data_iter()
        for _smile in pm2_iter:
            fobj.write("%s\n" % _smile.strip())


def create_vocabulary(
    data_path=os.path.join(FLAGS.data_dir, DATA_TMP_FILE),
    vocab_path=os.path.join(FLAGS.data_dir, VOCAB_FILE)):
    """Create vocabulary.

    Return vocabulary and reverse vocabulary.
    """
    words = set()
    with open(data_path) as data_fobj:
        for _smile in data_fobj:
            _smile = _smile.strip()
            words.update(list(_smile))
    with open(vocab_path, "w") as vocab_fobj:
        vocab_fobj.write("\n".join(list(words)))
    vocab = zip(list(words), range(1, len(words)+1))
    return vocab, list(words)


def get_vocabulary(vocab_path=os.path.join(FLAGS.data_dir, VOCAB_FILE)):
    """Get Vocabulary.

    Return the same as create_vocabulary.
    """
    with oepn(vocab_path) as vocab_fobj:
        words = vocab_fobj.read().split("\n")
        vocab = zip(list(words), range(1, len(words)+1))
    return vocab, words


def main(unused_argv):
    """Main function for this file."""
    build_tmp_file()
    vocab, words = create_vocabulary()


if __name__ == "__main__":
    tf.app.run()