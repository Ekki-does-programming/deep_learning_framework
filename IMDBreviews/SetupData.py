# Downloads and extracts the IMDB dataset and GloVe embeddings expected by
# TrainSentimentRNN.py / TrainSentimentLSTM.py (DATA_DIR="aclImdb",
# GLOVE_PATH="glove.6B.50d.txt"). Safe to re-run — skips anything already present.

import os
import sys
import tarfile
import urllib.request
import zipfile

IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
IMDB_ARCHIVE = "aclImdb_v1.tar.gz"
IMDB_DIR = "aclImdb"

GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_ARCHIVE = "glove.6B.zip"
GLOVE_FILE = "glove.6B.50d.txt"


def _report_progress(block_num, block_size, total_size):
    if total_size <= 0:
        return
    done = min(block_num * block_size, total_size)
    pct = done * 100 // total_size
    sys.stdout.write(f"\r  {done / 1e6:.1f} / {total_size / 1e6:.1f} MB ({pct}%)")
    sys.stdout.flush()


def _download(url, dest):
    if os.path.isfile(dest):
        print(f"{dest} already downloaded, skipping.")
        return
    print(f"Downloading {url} …")
    urllib.request.urlretrieve(url, dest, reporthook=_report_progress)
    print()


def setup_imdb():
    if os.path.isdir(IMDB_DIR):
        print(f"{IMDB_DIR}/ already exists, skipping IMDB setup.")
        return
    _download(IMDB_URL, IMDB_ARCHIVE)
    print(f"Extracting {IMDB_ARCHIVE} …")
    with tarfile.open(IMDB_ARCHIVE) as tar:
        tar.extractall(".", filter="data")
    os.remove(IMDB_ARCHIVE)


def setup_glove():
    if os.path.isfile(GLOVE_FILE):
        print(f"{GLOVE_FILE} already exists, skipping GloVe setup.")
        return
    _download(GLOVE_URL, GLOVE_ARCHIVE)
    print(f"Extracting {GLOVE_FILE} from {GLOVE_ARCHIVE} …")
    with zipfile.ZipFile(GLOVE_ARCHIVE) as z:
        z.extract(GLOVE_FILE, ".")
    os.remove(GLOVE_ARCHIVE)


if __name__ == "__main__":
    setup_imdb()
    setup_glove()
    print("Done.")
