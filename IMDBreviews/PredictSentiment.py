# Classify IMDB reviews with a trained sentiment model — either a random
# test-set review (sanity check) or a review one types oneself.

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import NeuralNetwork
from IMDBreviews.SentimentTrainer import get_data_layer

DATA_CACHE_PATH = os.path.join("trained", "imdb_data.pkl")
MODEL_PATHS = {
    "rnn": os.path.join("trained", "SentimentRNN"),
    "lstm": os.path.join("trained", "SentimentLSTM"),
}
LABEL_NAMES = ["negative", "positive"]

# must match the hyperparameters used by TrainSentimentRNN.py / TrainSentimentLSTM.py,
# only used if trained/imdb_data.pkl doesn't exist yet
EMBEDDING_DIM = 50
MAX_SEQ_LEN = 100
VOCAB_SIZE = 10_000
DATA_DIR = "aclImdb"
GLOVE_PATH = "glove.6B.50d.txt"


def _resolve_model_path(requested):
    if requested is not None:
        path = MODEL_PATHS[requested]
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No trained model at {path!r} — train it first.")
        return path
    for name in ("lstm", "rnn"):
        path = MODEL_PATHS[name]
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "No trained model found in 'trained/' — run TrainSentimentRNN.py or TrainSentimentLSTM.py first."
    )


def _predict(net, encoded_review):
    probs = net.test(encoded_review)[0]
    predicted = int(np.argmax(probs))
    return predicted, probs[predicted]


def classify_sample_review(net, data_layer):
    idx = random.randrange(len(data_layer.test_data))
    text = data_layer.test_texts[idx]
    true_label = int(np.argmax(data_layer.test_labels[idx]))
    predicted, confidence = _predict(net, data_layer.test_data[idx])

    preview = text if len(text) <= 500 else text[:500] + " ..."
    print(f"\nReview:\n{preview}")
    print(f"\nActual   : {LABEL_NAMES[true_label]}")
    print(f"Predicted: {LABEL_NAMES[predicted]}  (confidence {confidence * 100:.1f}%)")


def classify_typed_review(net, data_layer):
    text = input("\nType a review: ").strip()
    if not text:
        print("Empty review, skipping.")
        return
    encoded = data_layer.encode_text(text)
    predicted, confidence = _predict(net, encoded)
    print(f"Predicted: {LABEL_NAMES[predicted]}  (confidence {confidence * 100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Classify IMDB reviews with a trained sentiment model.")
    parser.add_argument("--model", choices=["rnn", "lstm"], default=None,
                         help="Which trained model to load (default: lstm if available, else rnn)")
    args = parser.parse_args()

    model_path = _resolve_model_path(args.model)
    data_layer = get_data_layer(DATA_DIR, GLOVE_PATH, MAX_SEQ_LEN, VOCAB_SIZE, EMBEDDING_DIM, DATA_CACHE_PATH)

    print(f"Loading model from {model_path!r} ...")
    net = NeuralNetwork.load(model_path, data_layer)

    while True:
        print("\n1) Classify a random test review")
        print("2) Type your own review")
        print("3) Quit")
        choice = input("> ").strip()

        if choice == "1":
            classify_sample_review(net, data_layer)
        elif choice == "2":
            classify_typed_review(net, data_layer)
        elif choice in ("3", "q", "quit"):
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
