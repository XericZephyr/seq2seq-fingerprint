"""Seq2seq Model Extension in Seq2seq-fingerprint."""

from __future__ import print_function


import numpy as np
import tensorflow as tf

from tensorflow.models.rnn.translate import seq2seq_model


class Seq2SeqModel(seq2seq_model.Seq2SeqModel):

    def __init__(self, *args, **kwargs):
        """Initialize the sequential model for Seq2seq fingerprint."""
        super(self.__class__, self).__init__(*args, **kwargs)
        self.num_layers = args[4] 


    def _get_encoder_state_names(self, bucket_id):
        """Get names of encoder_state."""
        if bucket_id == 0:
            prefix = "model_with_buckets/embedding_attention_seq2seq"
        else:
            prefix = "model_with_buckets/embedding_attention_seq2seq_%d" % bucket_id
        if self.num_layers < 2:
            raise NotImplementedError("Cannot get state name for 1-layer RNN.")
        cell_prefix = "%s/RNN/MultiRNNCell_%d" % (prefix, self.buckets[bucket_id][0]-1)
        encoder_state_names = [
            "%s/Cell%d/%s/add:0" % (
                cell_prefix,
                cell_id,
                "GRUCell" # In the future, we might have LSTM support.
            ) for cell_id in xrange(self.num_layers)]
        return encoder_state_names


    def step(self, session, encoder_inputs, decoder_inputs, target_weights,
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

        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
        else:
            if output_encoder_states:
                 # No gradient norm, loss, outputs, encoder fixed vector.
                return None, outputs[0], outputs[1:1+decoder_size], outputs[1+decoder_size:]
            else:
                # No gradient norm, loss, outputs.
                return None, outputs[0], outputs[1:1+decoder_size] 
