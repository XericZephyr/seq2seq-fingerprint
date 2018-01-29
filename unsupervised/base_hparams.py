""" Hyper parameters"""
import tensorflow as tf

def build_base_hparams():
    """build hyper-parameters"""
    hparams = tf.contrib.training.HParams(dropout_rate=0.5,
                                          num_layers=3,
                                          size=128,
                                          learning_rate=0.5,
                                          learning_rate_decay_factor=0.99,
                                          buckets=[[30, 30], [60, 60], [90, 90]],
                                          target_vocab_size=41,
                                          batch_size=256,
                                          source_vocab_size=41,
                                          max_gradient_norm=5.0)
    return hparams
