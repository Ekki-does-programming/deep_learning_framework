import numpy as np
import os
import pickle
import re


class IMDBData:
    """
    Data layer for the IMDB sentiment dataset.

    The RNN & LSTM treat the batch axis as time axis. Thus, each call to next()
    returns a single review as (T, embedding_dim) plus a (1,2) one-hot label,
    i.e. the effective batch size is 1.

    Directory layout (from aclImdb_v1.tar.gz):
        aclImdb/
            train/pos/*.txt
            train/neg/*.txt
            test/pos/*.txt
            test/neg/*.txt

    Parameters
    ----------
    data_dir         : str   - path to the extracted aclImdb folder
    glove_path       : str   - path to the .txt GloVe file
    max_seq_len      : int   - pad / truncate every review to this many words
    vocab_size       : int   - keep only the top-N most frequent words
    embedding_dim    : int   - must match the GloVe file you chose (50 / 100 / 200 / 300)
    """

    def __init__(
        self,
        data_dir: str = "aclImdb",
        glove_path: str = "glove.6B.50d.txt",
        max_seq_len: int = 100,
        vocab_size: int = 10_000,
        embedding_dim: int = 50,
    ):
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim

        print("Loading IMDB reviews ...")
        train_texts, train_labels = self._load_split(data_dir, "train")
        test_texts,  test_labels  = self._load_split(data_dir, "test")

        print("Building vocabulary ...")
        self.word2idx = self._build_vocab(train_texts, vocab_size)

        print("Loading GloVe embeddings ...")
        self.embedding_matrix = self._load_glove(glove_path, self.word2idx, embedding_dim)

        print("Encoding sequences ...")
        # Each entry: (T, D) array
        self.train_data   = [self._encode_one(t) for t in train_texts]   # (N, T, D)
        self.train_labels = self._one_hot(train_labels) # (N, 2)
        self.test_data    = [self._encode_one(t) for t in test_texts]
        self.test_labels  = self._one_hot(test_labels)
        self.test_texts   = test_texts  # kept for display in prediction demos

        self._cursor = 0
        self._order  = np.arange(len(self.train_data))
        np.random.shuffle(self._order)

        print(
            f"Ready — {len(self.train_data)} train / {len(self.test_data)} test samples, "
            f"sequence length {max_seq_len}, embedding dim {embedding_dim}."
        )

    # ------------------------------------------------------------------
    # Public interface (mirrors MNISTData)
    # ------------------------------------------------------------------

    def next(self):
        """Return (data, labels) for a single review."""
        idx = self._order[self._cursor]
        self._cursor += 1
        if self._cursor >= len(self.train_data):
            self._cursor = 0
            np.random.shuffle(self._order)
        return self.train_data[idx], self.train_labels[[idx]]

    def get_test_set(self):
        """Return the full test set as (data, labels)."""
        return self.test_data, self.test_labels

    def encode_text(self, text):
        """Encode an arbitrary (e.g. user-typed) review with this instance's
        vocabulary/embedding matrix, for prediction on new text."""
        return self._encode_one(text)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_split(data_dir: str, split: str):
        texts, labels = [], []
        for label, sentiment in enumerate(["neg", "pos"]):
            folder = os.path.join(data_dir, split, sentiment)
            for fname in sorted(os.listdir(folder)):
                if not fname.endswith(".txt"):
                    continue
                with open(os.path.join(folder, fname), encoding="utf-8") as f:
                    texts.append(f.read())
                labels.append(label)
        return texts, labels

    @staticmethod
    def _tokenize(text: str):
        text = text.lower()
        text = re.sub(r"<[^>]+>", " ", text)          # strip HTML tags
        text = re.sub(r"[^a-z0-9' ]", " ", text)      # keep only letters/digits/apostrophes
        return text.split()

    def _build_vocab(self, texts, vocab_size):
        counts = {}
        for text in texts:
            for word in self._tokenize(text):
                counts[word] = counts.get(word, 0) + 1
        top_words = sorted(counts, key=counts.get, reverse=True)[:vocab_size]
        return {word: idx + 2 for idx, word in enumerate(top_words)}

    @staticmethod
    def _load_glove(glove_path, word2idx, embedding_dim):
        """Build an embedding matrix (vocab_size+2, D) from a GloVe .txt file."""
        vocab_size = len(word2idx) + 2
        matrix = np.zeros((vocab_size, embedding_dim), dtype=np.float32)
        found = 0
        with open(glove_path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                word = parts[0]
                if word in word2idx:
                    matrix[word2idx[word]] = np.array(parts[1:], dtype=np.float32)
                    found += 1
        print(f"  GloVe: matched {found}/{len(word2idx)} vocabulary words.")
        return matrix  # shape (V, D)

    def _encode_one(self, text):
        """Convert a single review string to a padded numpy array (T, embedding_dim)."""
        T, D = self.max_seq_len, self.embedding_dim
        out = np.zeros((T, D), dtype=np.float32)
        tokens = self._tokenize(text)[:T]
        for t, word in enumerate(tokens):
            idx = self.word2idx.get(word, 1)  # 1 = <UNK>
            out[t] = self.embedding_matrix[idx]
        return out

    @staticmethod
    def _one_hot(labels):
        out = np.zeros((len(labels), 2), dtype=np.float32)
        for i, lbl in enumerate(labels):
            out[i, lbl] = 1.0
        return out


def save(filename, data_layer):
    # pickle dump to file — avoids re-reading aclImdb + rescanning GloVe on every run
    with open(filename, "wb") as file:
        pickle.dump(data_layer, file)


def load(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)
