"""Decode fingerprint for the format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import tensorflow as tf
import smile as sm

from unsupervised.seq2seq_model import FingerprintFetcher

with sm.app.flags.Subcommand("sample", dest="action"):
    sm.app.flags.DEFINE_string("data_path", "", "Data path of the sample.", required=True)
    sm.app.flags.DEFINE_integer("sample_size", 100, "Sample size from the data file.")

sm.app.flags.DEFINE_string("model_dir", "", "model path of the seq2seq fingerprint.",
                           required=True)
sm.app.flags.DEFINE_string("vocab_path", "", "Vocabulary path of the seq2seq fingerprint.",
                           required=True)

FLAGS = sm.app.flags.FLAGS

def sample_smiles(data_path, sample_size):
    """Sample several sentences."""
    samples = set()
    with open(data_path) as fobj:
        lines = [_line for _line in fobj.readlines() if len(_line.strip())]
    while len(samples) < sample_size:
        samples.add(random.randrange(len(lines)))
    return [lines[index].strip() for index in list(samples)]

def sample_decode():
    """Sample some samples from data file and print out the recovered string."""
    with tf.Session() as sess:
        sampled_smiles = sample_smiles(FLAGS.data_path, FLAGS.sample_size)
        fetcher = FingerprintFetcher(FLAGS.model_dir, FLAGS.vocab_path, sess)
        exact_match_num = 0
        for smile in sampled_smiles:
            _, output_smile = fetcher.decode(smile)
            if output_smile == smile:
                exact_match_num += 1
            print(": %s\n> %s\n" % (smile, output_smile))
        print("Exact match count: %d/%d" % (exact_match_num, len(sampled_smiles)))

def main(_):
    """Entry function for the script."""
    if FLAGS.action == "sample":
        sample_decode()
    else:
        print("Unsupported action: %s" % FLAGS.action)

if __name__ == "__main__":
    sm.app.run()
