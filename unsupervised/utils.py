"""Docstring for this file."""

from __future__ import division
from __future__ import print_function

from tensorflow.models.rnn.translate.data_utils import (
    create_vocabulary, initialize_vocabulary)

MAX_SMILE_VOCAB_TOKEN = 10000

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
