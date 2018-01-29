"""Train fingerprint."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys
import time

import tensorflow as tf
import smile as sm
import numpy as np

from unsupervised import seq2seq_model
from unsupervised.utils import EOS_ID, PAD_ID
from unsupervised.base_hparams import build_base_hparams

with sm.app.flags.Subcommand("build", dest="action"):
    sm.app.flags.DEFINE_string("model_dir", "", "model path of the seq2seq fingerprint.",
                               required=True)
    sm.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")
    sm.app.flags.DEFINE_integer("size", 128, "Size of each model layer.")
    sm.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
    sm.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                              "Learning rate decays by this much.")
    sm.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
    sm.app.flags.DEFINE_float("dropout_rate", 0.5, "dropout rate")
    sm.app.flags.DEFINE_string("buckets", "[[30, 30], [60, 60], [90, 90]]", "buckets")
    sm.app.flags.DEFINE_integer("target_vocab_size", 41, "target vocab size")
    sm.app.flags.DEFINE_integer("batch_size", 256, "dropout rate")
    sm.app.flags.DEFINE_integer("source_vocab_size", 41, "source vocab size")



with sm.app.flags.Subcommand("train", dest="action"):
    sm.app.flags.DEFINE_string("model_dir", "", "model path of the seq2seq fingerprint.",
                               required=True)
    sm.app.flags.DEFINE_string("train_data", "", "train_data for seq2seq fp train.",
                               required=True)
    sm.app.flags.DEFINE_string("test_data", "", "test data path of the seq2seq fp eval.",
                               required=True)
    sm.app.flags.DEFINE_integer("batch_size", 128,
                                "Batch size to use during training.")
    sm.app.flags.DEFINE_integer("gpu", 0,
                                "GPU device to use, default: 0.")
    sm.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                                "How many training steps to do per checkpoint.")
    sm.app.flags.DEFINE_string("summary_dir", "", "Summary dir.")



FLAGS = sm.app.flags.FLAGS

def build_hparams():
    """build model.json using hyper-parameters"""
    hparams = build_base_hparams()
    model_file = os.path.join(FLAGS.model_dir, "model.json")
    with open(model_file, "w") as fobj:
        fobj.write(hparams.to_json())


def read_data(source_path, buckets, max_size=None):
    """Read data from source and target files and put into buckets.

    Args:
        source_path: path to the files with token-ids for the source language.
        max_size: maximum number of lines to read, all other will be ignored;
            if 0 or None, data files will be read completely (no limit).

    Returns:
        data_set: a list of length len(_buckets); data_set[n] contains a list of
            (source, target) pairs read from the provided data files that fit
            into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
            len(target) < _buckets[n][1]; source and target are lists of token-ids.
    """
    data_set = [[] for _ in buckets]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        source = source_file.readline()
        counter = 0
        while source and (not max_size or counter < max_size):
            counter += 1
            if counter % 100000 == 0:
                print("  reading data line %d" % counter)
                sys.stdout.flush()
            source_ids = [int(x) for x in source.split()]
            target_ids = [int(x) for x in source.split()]
            target_ids.append(EOS_ID)
            for bucket_id, (source_size, target_size) in enumerate(buckets):
                if len(source_ids) < source_size and len(target_ids) < target_size:
                    data_set[bucket_id].append([source_ids, target_ids])
                    break
            source = source_file.readline()
    return data_set

def train(train_data, test_data): # pylint: disable=too-many-locals
    """Train script."""
    model_dir = FLAGS.model_dir
    batch_size = FLAGS.batch_size

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        with tf.device("/gpu:%d" % FLAGS.gpu):
            # Create model.
            model = seq2seq_model.Seq2SeqModel.load_model_from_dir(
                model_dir, False, sess=sess)
        buckets = model.buckets
        model.batch_size = batch_size

        # Read data into buckets and compute their sizes.
        print("Reading train data from %s..." % train_data)
        train_set = read_data(train_data, buckets)
        print("Reading test data from %s..." % test_data)
        test_set = read_data(test_data, buckets)

        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(buckets))]
        train_total_size = float(sum(train_bucket_sizes))
        train_bucket_prob = [size / train_total_size for size in train_bucket_sizes]

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        if FLAGS.summary_dir:
            train_writer = tf.summary.FileWriter(os.path.join(FLAGS.summary_dir, "train"),
                                                 sess.graph)
            test_writer = tf.summary.FileWriter(os.path.join(FLAGS.summary_dir, "test"),
                                                sess.graph)

        # TODO(zhengxu): In the future, we should move this to seq2seq model class.
        def get_em_acc_op(bucket_id):
            """Create a em_acc_op."""
            with tf.name_scope("EMAcc_%d" % bucket_id):
                input_ph = tf.placeholder(tf.int64, shape=(None, batch_size))
                output_ph = tf.placeholder(tf.float32, shape=(
                    None, batch_size, model.target_vocab_size))
                input_op = tf.reverse_v2(input_ph, axis=[0])
                output_op = tf.argmax(output_ph, axis=2)
                def replace_eos_with_pad(in_seq):
                    """Replace all tokens after EOS in sequence with PAD."""
                    out_seq = in_seq.copy()
                    for idx in xrange(in_seq.shape[-1]):
                        eos_list = in_seq[:, idx].reshape(in_seq.shape[0]).tolist()
                        eos_idx = eos_list.index(EOS_ID) if EOS_ID in eos_list else -1
                        out_seq[eos_idx:, idx] = PAD_ID
                    return out_seq

                eos_op = tf.py_func(replace_eos_with_pad, [output_op], tf.int64)
                equal_op = tf.equal(tf.reduce_sum(tf.abs(input_op - eos_op), axis=0), 0)
                em_acc_op = tf.reduce_mean(tf.cast(equal_op, tf.float32), axis=0)
                summary_op = tf.summary.scalar("EMAccSummary", em_acc_op)
            return input_ph, output_ph, em_acc_op, summary_op

        test_summary_ops = [get_em_acc_op(bucket_id) for bucket_id in xrange(len(buckets))]

        while True:
            # Choose a bucket according to data distribution. We pick a random number
            # in [0, 1] and use the corresponding interval in train_buckets_scale.
            bucket_id = np.random.choice(len(train_bucket_prob), p=train_bucket_prob)

            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                train_set, bucket_id)
            _, step_loss, summary = model.step(sess, encoder_inputs, decoder_inputs,
                                               target_weights, bucket_id, False)
            if FLAGS.summary_dir:
                train_writer.add_summary(summary, model.global_step.eval())
            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1


            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % FLAGS.steps_per_checkpoint == 0:
                # Print statistics for the previous epoch.
                perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                print ("global step %d learning rate %.4f step-time %.6f perplexity "
                       "%.6f" % (model.global_step.eval(), model.learning_rate_op.eval(),
                                 step_time, perplexity))
                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                model.save_model_to_dir(model_dir, sess=sess)
                step_time, loss = 0.0, 0.0
                # Run evals on development set and print their perplexity.
                for bucket_id in xrange(len(buckets)):
                    length_test_set = len(test_set[bucket_id])
                    if length_test_set == 0:
                        print("  eval: empty bucket %d" % (bucket_id))
                        continue
                    encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                        test_set, bucket_id)
                    _, eval_loss, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                                             target_weights, bucket_id, True)
                    input_ph, output_ph, em_acc_op, summary_op = test_summary_ops[bucket_id]
                    em_acc, summary = sess.run(
                        [em_acc_op, summary_op],
                        feed_dict={
                            input_ph: np.array(encoder_inputs),
                            output_ph: np.array(output_logits)})
                    if FLAGS.summary_dir:
                        test_writer.add_summary(summary, model.global_step.eval())

                    eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float("inf")
                    print("  eval: bucket %d perplexity %.6f, em_acc %.6f" % (
                        bucket_id, eval_ppx, em_acc))
                sys.stdout.flush()


def main(_):
    """Entry function for the script."""
    if FLAGS.action == "build":
        build_hparams()
    elif FLAGS.action == "train":
        train(FLAGS.train_data, FLAGS.test_data)
    else:
        print("Unsupported action: %s" % FLAGS.action)

if __name__ == "__main__":
    sm.app.run()
