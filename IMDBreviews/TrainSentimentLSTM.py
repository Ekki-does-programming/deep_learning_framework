# Training script for the sentiment LSTM on the IMDB dataset.
# Mirrors TrainSentimentRNN.py as closely as possible — both delegate the
# actual train/evaluate/plot logic to SentimentTrainer.train_and_evaluate().

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from IMDBreviews.SentimentTrainer import train_and_evaluate

# ------------------------------------------------------------------
# Hyperparameters — tweak these first if results are unsatisfactory
# ------------------------------------------------------------------
ITERATIONS    = 25_000   # each iteration = one review
EMBEDDING_DIM = 50       # must match your GloVe file (50d here)
MAX_SEQ_LEN   = 100      # words per review (longer = slower, more context)
VOCAB_SIZE    = 10_000
HIDDEN_SIZE   = 64

DATA_DIR      = os.path.join(SCRIPT_DIR, "aclImdb")           # path to extracted IMDB folder
GLOVE_PATH    = os.path.join(SCRIPT_DIR, "glove.6B.50d.txt")  # path to GloVe embeddings file
SAVE_PATH     = os.path.join(SCRIPT_DIR, "trained", "SentimentLSTM")

train_and_evaluate(
    cell_type="lstm",
    save_path=SAVE_PATH,
    iterations=ITERATIONS,
    embedding_dim=EMBEDDING_DIM,
    max_seq_len=MAX_SEQ_LEN,
    vocab_size=VOCAB_SIZE,
    hidden_size=HIDDEN_SIZE,
    data_dir=DATA_DIR,
    glove_path=GLOVE_PATH,
)
