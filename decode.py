"""Decode fingerprint for the format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import tensorflow as tf
import smile as sm
from smile import logging
from unsupervised.seq2seq_model import FingerprintFetcher

with sm.app.flags.Subcommand("sample", dest="action"):
    sm.app.flags.DEFINE_string("model_dir", "", "model path of the seq2seq fingerprint.",
                               required=True)
    sm.app.flags.DEFINE_string("vocab_path", "", "Vocabulary path of the seq2seq fingerprint.",
                               required=True)
    sm.app.flags.DEFINE_string("data_path", "", "Data path of the sample.", required=True)
    sm.app.flags.DEFINE_integer("sample_size", 100, "Sample size from the data file.")

with sm.app.flags.Subcommand("fp", dest="action"):
    sm.app.flags.DEFINE_string("model_dir", "", "model path of the seq2seq fingerprint.",
                               required=True)
    sm.app.flags.DEFINE_string("vocab_path", "", "Vocabulary path of the seq2seq fingerprint.",
                               required=True)
    sm.app.flags.DEFINE_string("data_path", "", "Required data path.", required=True)
    sm.app.flags.DEFINE_string("output_path", "", "Output path of the sample.", required=True)


FLAGS = sm.app.flags.FLAGS

def sample_smiles(data_path, sample_size):
    """Sample several sentences."""
    samples = set()
    with open(data_path) as fobj:
        lines = [_line for _line in fobj.readlines() if len(_line.strip())]
    data_size = len(lines)
    if data_size < sample_size:
        sample_size_ori = sample_size
        sample_size = data_size
        logging.warning("sample size (%d) is too large, "
                        "data size (%d) is used instead as the sample size"
                        % (sample_size_ori, data_size))
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

def read_smiles(data_file):
    """Read all smile from a line-splitted file."""
    with open(data_file) as fobj:
        out_smiles = [_line.strip() for _line in fobj if _line.strip()]
    return out_smiles

def fp_decode():
    """Decode ALL samples from the given data file and output to file."""
    with tf.Session() as sess, open(FLAGS.output_path, "w") as fout:
        all_smiles = read_smiles(FLAGS.data_path)
        fetcher = FingerprintFetcher(FLAGS.model_dir, FLAGS.vocab_path, sess)
        exact_match_num = 0
        for idx, smile in enumerate(all_smiles):
            seq2seq_fp, output_smile = fetcher.decode(smile)
            if output_smile == smile:
                exact_match_num += 1
            fout.write(" ".join([str(fp_bit) for fp_bit in seq2seq_fp]) + "\n")
            if idx % 200 == 0 and idx:
                print("Progress: %d/%d" % (idx, len(all_smiles)))
        print("Exact match count: %d/%d" % (exact_match_num, len(all_smiles)))

def main(_):
    """Entry function for the script."""
    if FLAGS.action == "sample":
        sample_decode()
    elif FLAGS.action == "fp":
        fp_decode()
    else:
        print("Unsupported action: %s" % FLAGS.action)

if __name__ == "__main__":
    sm.app.run()
