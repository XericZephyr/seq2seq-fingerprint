"""Seq2seq Model Extension in Seq2seq-fingerprint."""
# pylint: disable=invalid-name

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import random

import numpy as np
import tensorflow as tf

from .base_hparams import build_base_hparams
from .utils import (initialize_vocabulary, sentence_to_token_ids, smile_tokenizer, EOS_ID, PAD_ID,
                    GO_ID)

class Seq2SeqModel(object): # pylint: disable=too-many-instance-attributes
    """Customized seq2seq model for fingerprint method."""
    MODEL_PARAMETER_FIELDS = [
        # Feedforward parameters.
        "source_vocab_size", "target_vocab_size", "buckets", "size", "num_layers", "dropout_rate",
        # Training parameters.
        "max_gradient_norm", "batch_size", "learning_rate", "learning_rate_decay_factor"
    ]

    def __init__(self, # pylint: disable=too-many-locals, too-many-arguments, super-init-not-called
                 hparams,
                 forward_only=False,
                 use_lstm=False,
                 num_samples=512,
                 dtype=tf.float32):
        """Create the model.
        Args:
            source_vocab_size: size of the source vocabulary.
            target_vocab_size: size of the target vocabulary.
            buckets: a list of pairs (I, O), where I specifies maximum input length
                that will be processed in that bucket, and O specifies maximum output
                length. Training instances that have inputs longer than I or outputs
                longer than O will be pushed to the next bucket and padded accordingly.
                We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
            size: number of units in each layer of the model.
            num_layers: number of layers in the model.
            max_gradient_norm: gradients will be clipped to maximally this norm.
            batch_size: the size of the batches used during training;
                the model construction is independent of batch_size, so it can be
                changed after initialization if this is convenient, e.g., for decoding.
            learning_rate: learning rate to start with.
            learning_rate_decay_factor: decay learning rate by this much when needed.
            use_lstm: if true, we use LSTM cells instead of GRU cells.
            num_samples: number of samples for sampled softmax.
            forward_only: if set, we do not construct the backward pass in the model.
            dtype: the data type to use to store internal variables.
        """
        self.source_vocab_size = hparams.source_vocab_size
        self.target_vocab_size = hparams.target_vocab_size
        self.buckets = hparams.buckets
        self.size = hparams.size
        self.num_layers = hparams.num_layers
        self.max_gradient_norm = hparams.max_gradient_norm
        self.batch_size = hparams.batch_size
        self.learning_rate = hparams.learning_rate
        self.learning_rate_decay_factor = hparams.learning_rate_decay_factor
        self.learning_rate_op = tf.Variable(
            float(self.learning_rate), trainable=False, dtype=dtype)
        self.learning_rate_decay_op = self.learning_rate_op.assign(
            self.learning_rate_op * hparams.learning_rate_decay_factor)
        self.dropout_rate = hparams.dropout_rate
        self.global_step = tf.Variable(0, trainable=False)

        size = hparams.size
        buckets = hparams.buckets
        dropout_rate = hparams.dropout_rate
        num_layers = hparams.num_layers

        # If we use sampled softmax, we need an output projection.
        output_projection = None
        softmax_loss_function = None
        # Sampled softmax only makes sense if we sample less than vocabulary size.
        if num_samples > 0 and num_samples < self.target_vocab_size:
            w_t = tf.get_variable("proj_w", [self.target_vocab_size, hparams.size], dtype=dtype)
            w = tf.transpose(w_t)
            b = tf.get_variable("proj_b", [self.target_vocab_size], dtype=dtype)
            output_projection = (w, b)

            def sampled_loss(labels, logits):
                """Sampleed loss function."""
                labels = tf.reshape(labels, [-1, 1])
                # We need to compute the sampled_softmax_loss using 32bit floats to
                # avoid numerical instabilities.
                local_w_t = tf.cast(w_t, tf.float32)
                local_b = tf.cast(b, tf.float32)
                local_inputs = tf.cast(logits, tf.float32)
                return tf.cast(
                    tf.nn.sampled_softmax_loss(local_w_t, local_b, labels,
                                               local_inputs, num_samples,
                                               self.target_vocab_size), dtype)
            softmax_loss_function = sampled_loss

        # Create the internal multi-layer cell for our RNN.
        def single_cell():
            """internal single cell for RNN"""
            if use_lstm:
                ret = tf.contrib.rnn.BasicLSTMCell(hparams.size)
            else:
                ret = tf.contrib.rnn.GRUCell(hparams.size)
            ret = tf.nn.rnn_cell.DropoutWrapper(
                ret,
                input_keep_prob=dropout_rate,
                output_keep_prob=dropout_rate)
            return ret
        # The seq2seq function: we use embedding for the input and attention.
        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            """Sequence to sequence function."""
            cell = single_cell()
            if num_layers > 1:
                cell = tf.contrib.rnn.MultiRNNCell(
                    [single_cell() for _ in xrange(num_layers)])
            return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                encoder_inputs,
                decoder_inputs,
                cell,
                num_encoder_symbols=hparams.source_vocab_size,
                num_decoder_symbols=hparams.target_vocab_size,
                embedding_size=size,
                output_projection=output_projection,
                feed_previous=do_decode,
                dtype=dtype)

        # Feeds for inputs.
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                      name="encoder{0}".format(i)))
        for i in xrange(buckets[-1][1] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                      name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(dtype, shape=[None],
                                                      name="weight{0}".format(i)))

        # Our targets are decoder inputs shifted by one.
        targets = [self.decoder_inputs[i + 1]
                   for i in xrange(len(self.decoder_inputs) - 1)]

        # Training outputs and losses.
        if forward_only:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, targets,
                self.target_weights, buckets, lambda x, y: seq2seq_f(x, y, True),
                softmax_loss_function=softmax_loss_function)
            # If we use output projection, we need to project outputs for decoding.
            if output_projection is not None:
                for b in xrange(len(buckets)):
                    self.outputs[b] = [
                        tf.matmul(output, output_projection[0]) + output_projection[1]
                        for output in self.outputs[b]
                    ]
        else:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, targets,
                self.target_weights, buckets,
                lambda x, y: seq2seq_f(x, y, False),
                softmax_loss_function=softmax_loss_function)

        # Gradients and SGD update operation for training the model.
        params = tf.trainable_variables()
        self.summary_ops = []
        if not forward_only:
            self.gradient_norms = []
            self.updates = []
            lr_summary_op = tf.summary.scalar("learning rate", self.learning_rate_op)
            opt = tf.train.GradientDescentOptimizer(self.learning_rate_op)
            for b in xrange(len(buckets)):
                gradients = tf.gradients(self.losses[b], params)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                                 hparams.max_gradient_norm)
                self.gradient_norms.append(norm)
                self.updates.append(opt.apply_gradients(
                    zip(clipped_gradients, params), global_step=self.global_step))
                self.summary_ops.append(tf.summary.merge([
                    tf.summary.scalar("global_norm_%d" % b, norm),
                    tf.summary.scalar("loss_%d" % b, self.losses[b]),
                    lr_summary_op
                ]))
        self.saver = tf.train.Saver(tf.global_variables())

#
#   Model load and save.
#

    @classmethod
    def load_model_from_files(cls, model_file, checkpoint_dir, forward_only, sess=None):
        """Load model from file."""
        hparams = build_base_hparams()
        print("Loading seq2seq model definition from %s..." % model_file)
        with open(model_file, "r") as fobj:
            model_dict = json.load(fobj)
        model_dict["buckets"] = [tuple(_bucket) for _bucket in model_dict["buckets"]]
        hparams.set_from_map(model_dict)
        model = cls(hparams, forward_only)
        # Load model weights.
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        sess = sess or tf.get_default_session()
        if ckpt:
            print("Loading model weights from checkpoint_dir: %s" % checkpoint_dir)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Initialize fresh parameters...")
            sess.run(tf.global_variables_initializer())
        return model

    @classmethod
    def load_model_from_dir(cls, train_dir, forward_only, sess=None):
        """Load model definition from train_dir/model.json and train_dir/weights."""
        model_file = os.path.join(train_dir, "model.json")
        checkpoint_dir = os.path.join(train_dir, "weights/")
        return cls.load_model_from_files(model_file, checkpoint_dir, forward_only, sess)

    def save_model_to_files(self, model_file, checkpoint_file, sess=None, verbose=False):
        """Save all the model hyper-parameters to a json file."""
        if verbose:
            print("Save model defintion to %s..." % model_file)
        model_dict = {key: getattr(self, key) for key in self.MODEL_PARAMETER_FIELDS}
        with open(model_file, "w") as fobj:
            json.dump(model_dict, fobj)
        checkpoint_dir = os.path.dirname(checkpoint_file)
        if os.path.exists(checkpoint_dir):
            if verbose:
                print("Save weights to %s..." % checkpoint_file)
            sess = sess or tf.get_default_session()
            self.saver.save(sess, checkpoint_file, global_step=self.global_step)
        elif verbose:
            print("Skip save weights to %s since the dir does not exist." % checkpoint_dir)

    def save_model_to_dir(self, train_dir, sess=None, verbose=False):
        """Save model definition and weights to train_dir/model.json and train_dir/checkpoints/"""
        model_file = os.path.join(train_dir, "model.json")
        checkpoint_dir = os.path.join(train_dir, "weights")
        checkpoint_file = os.path.join(checkpoint_dir, "weights-ckpt")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.save_model_to_files(model_file, checkpoint_file, sess=sess, verbose=verbose)

    def _get_encoder_state_names(self, bucket_id):
        """Get names of encoder_state."""
        if bucket_id == 0:
            prefix = "model_with_buckets/embedding_attention_seq2seq"
        else:
            prefix = "model_with_buckets/embedding_attention_seq2seq_%d" % bucket_id
        if self.num_layers < 2:
            raise NotImplementedError("Cannot get state name for 1-layer RNN.")
        cell_prefix = ("%s/rnn/rnn/embedding_wrapper/embedding_wrapper/"
                       "multi_rnn_cell" % prefix)
        n = self.buckets[bucket_id][0]-1
        encoder_state_names = [
            "%s/cell_%d/cell_%d/%s/add%s:0" % (
                cell_prefix,
                cell_id,
                cell_id,
                "gru_cell", # In the future, we might have LSTM support.
                "_%d" % n if n > 0 else ""
            ) for cell_id in xrange(self.num_layers)]
        return encoder_state_names


    def step(self, session, encoder_inputs, decoder_inputs, target_weights, # pylint: disable=too-many-locals, too-many-arguments, too-many-branches, arguments-differ
             bucket_id, forward_only, output_encoder_states=False):
        """Run a step of the model feeding the given inputs.

        Args:
            session: tensorflow session to use.
            encoder_inputs: list of numpy int vectors to feed as encoder inputs.
            decoder_inputs: list of numpy int vectors to feed as decoder inputs.
            target_weights: list of numpy float vectors to feed as target weights.
            bucket_id: which bucket of the model to use.
            forward_only: whether to do the backward step or only forward.

        Returns:
            A triple consisting of gradient norm (or None if we did not do backward),
            average perplexity, and the outputs.

        Raises:
            ValueError: if length of encoder_inputs, decoder_inputs, or
                target_weights disagrees with bucket size for the specified bucket_id.
        """
        # Check if the sizes match.
        encoder_size, decoder_size = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(encoder_inputs), encoder_size))
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(decoder_inputs), decoder_size))
        if len(target_weights) != decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket,"
                             " %d != %d." % (len(target_weights), decoder_size))

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        for l in xrange(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in xrange(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        # Since our targets are decoder inputs shifted by one, we need one more.
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        # Output feed: depends on whether we do a backward step or not.
        if not forward_only:
            output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                           self.gradient_norms[bucket_id],  # Gradient norm.
                           self.losses[bucket_id]]  # Loss for this batch.
        else:
            output_feed = [self.losses[bucket_id]]  # Loss for this batch.
            for l in xrange(decoder_size):          # Output logits.
                output_feed.append(self.outputs[bucket_id][l])
            if output_encoder_states:
                default_graph = tf.get_default_graph()
                state_names = self._get_encoder_state_names(bucket_id)
                for state_name in state_names:
                    var = default_graph.get_tensor_by_name(state_name)
                    output_feed.append(var)

        if not forward_only:
            outputs, summary = session.run([output_feed, self.summary_ops[bucket_id]], input_feed)
            return outputs[1], outputs[2], summary  # Gradient norm, loss, no outputs.
        outputs = session.run(output_feed, input_feed)
        if output_encoder_states:
            # No gradient norm, loss, outputs, encoder fixed vector.
            return None, outputs[0], outputs[1:1+decoder_size], outputs[1+decoder_size:]
        # No gradient norm, loss, outputs.
        return None, outputs[0], outputs[1:1+decoder_size]


    def get_batch(self, data, bucket_id): # pylint: disable=too-many-locals
        """Get a random batch of data from the specified bucket, prepare for step.

        To feed data in step(..) it must be a list of batch-major vectors, while
        data here contains single length-major cases. So the main logic of this
        function is to re-index data cases to be in the proper format for feeding.

        Args:
            data: a tuple of size len(self.buckets) in which each element contains
                lists of pairs of input and output data that we use to create a batch.
            bucket_id: integer, which bucket to get the batch for.

        Returns:
            The triple (encoder_inputs, decoder_inputs, target_weights) for
            the constructed batch that has the proper format to call step(...) later.
        """
        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs, decoder_inputs = [], []

        # Get a random batch of encoder and decoder inputs from data,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        for _ in xrange(self.batch_size):
            encoder_input, decoder_input = random.choice(data[bucket_id])

            # Encoder inputs are padded and then reversed.
            encoder_pad = [PAD_ID] * (encoder_size - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

            # Decoder inputs get an extra "GO" symbol, and are padded then.
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([GO_ID] + decoder_input +
                                  [PAD_ID] * decoder_pad_size)

        # Now we create batch-major vectors from the data selected above.
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in xrange(encoder_size):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                          for batch_idx in xrange(self.batch_size)], dtype=np.int32))

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in xrange(decoder_size):
            batch_decoder_inputs.append(
                np.array([decoder_inputs[batch_idx][length_idx]
                          for batch_idx in xrange(self.batch_size)], dtype=np.int32))

            # Create target_weights to be 0 for targets that are padding.
            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for batch_idx in xrange(self.batch_size):
                # We set weight to 0 if the corresponding target is a PAD symbol.
                # The corresponding target is decoder_input shifted by 1 forward.
                if length_idx < decoder_size - 1:
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == decoder_size - 1 or target == PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)
        return batch_encoder_inputs, batch_decoder_inputs, batch_weights


class FingerprintFetcher(object):
    """Seq2seq fingerprint fetcher for the seq2seq fingerprint."""

    def __init__(self, model_dir, vocab_path, sess=None):
        """Initialize a fingerprint fetcher for the seq2seq-fingerprint."""
        self.model_dir = model_dir
        self.vocab_path = vocab_path

        # Load tensorflow model
        self.model = Seq2SeqModel.load_model_from_dir(self.model_dir, True, sess)
        self.model.batch_size = 1

        # Load vocabulary.
        self.vocab, self.rev_vocab = initialize_vocabulary(self.vocab_path)

    def get_bucket_id(self, token_ids):
        """Determine which bucket should the smile string be placed in."""
        _buckets = self.model.buckets
        bucket_id = len(_buckets) - 1
        for i, bucket in enumerate(_buckets):
            if bucket[0] >= len(token_ids):
                bucket_id = i
                break
        return bucket_id

    def decode(self, smile_string, sess=None): # pylint: disable=too-many-locals
        """Input a smile string and will output the fingerprint and predicted output."""
        token_ids = sentence_to_token_ids(
            tf.compat.as_bytes(smile_string), self.vocab,
            tokenizer=smile_tokenizer, normalize_digits=False)
        bucket_id = self.get_bucket_id(token_ids)
        # Get a 1-element batch to feed the sentence to the model.
        encoder_inputs, decoder_inputs, target_weights = self.model.get_batch(
            {bucket_id: [(token_ids, [])]}, bucket_id)
        # Get output logits for the sentence.
        sess = sess or tf.get_default_session()
        _, _, output_logits, fps = self.model.step(sess, encoder_inputs, decoder_inputs,
                                                   target_weights, bucket_id, True, True)
        # This is a greedy decoder - outputs are just argmaxes of output_logits.
        outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
        # If there is an EOS symbol in outputs, cut them at that point.
        if EOS_ID in outputs:
            outputs = outputs[:outputs.index(EOS_ID)]
        output_smile = "".join([tf.compat.as_str(self.rev_vocab[output]) for output in outputs])
        seq2seq_fp = np.concatenate(tuple([fp.flatten() for fp in fps]))
        # return the fingerprint and predicted smile.
        return seq2seq_fp, output_smile
