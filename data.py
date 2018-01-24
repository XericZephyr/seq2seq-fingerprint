"""Prepare data for seq2seq training."""

from __future__ import print_function

import smile as sm
from smile import flags
from unsupervised.data import build_vocab, translate_tokens


flags.DEFINE_string(
    "smi_path", "/smile/nfs/projects/nih_drug/data/logp/logp.smi", "smi data path.")
flags.DEFINE_string(
    "tmp_path", "", "Temporary data path. If none, a named temporary file will be used.")
flags.DEFINE_string(
    "vocab_path", "", "Vocabulary data_path.")
flags.DEFINE_string(
    "out_path", "", "Output token path.")
flags.DEFINE_bool(
    "build_vocab", False, "Trigger the action: False for translating only. "
    "If true, the script will build vocabulary and then translating.")

FLAGS = flags.FLAGS

def main(_):
    """Entry function for this script."""
    if FLAGS.build_vocab:
        build_vocab(FLAGS.smi_path, FLAGS.vocab_path, FLAGS.out_path, FLAGS.tmp_path)
    else:
        translate_tokens(FLAGS.smi_path, FLAGS.vocab_path, FLAGS.out_path, FLAGS.tmp_path)

if __name__ == "__main__":
    sm.app.run()
