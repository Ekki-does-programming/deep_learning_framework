# Shared train/evaluate/plot harness for the sentiment models (RNN and LSTM).
# TrainSentimentRNN.py and TrainSentimentLSTM.py each just set hyperparameters
# and call train_and_evaluate() with their cell_type.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

import NeuralNetwork
import IMDBreviews.IMDBData as IMDBDataModule
from IMDBreviews.IMDBData import IMDBData
from Models.SentimentModel import build
from Layers import Helpers


def get_data_layer(data_dir, glove_path, max_seq_len, vocab_size, embedding_dim, data_cache_path):
    if data_cache_path and os.path.isfile(data_cache_path):
        print("Loading cached IMDB data ...")
        return IMDBDataModule.load(data_cache_path)

    data_layer = IMDBData(
        data_dir=data_dir,
        glove_path=glove_path,
        max_seq_len=max_seq_len,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
    )
    if data_cache_path:
        os.makedirs(os.path.dirname(data_cache_path), exist_ok=True)
        IMDBDataModule.save(data_cache_path, data_layer)
    return data_layer


def train_and_evaluate(
    cell_type: str,
    save_path: str,
    iterations: int = 25_000,
    embedding_dim: int = 50,
    max_seq_len: int = 100,
    vocab_size: int = 10_000,
    hidden_size: int = 64,
    data_dir: str = "aclImdb",
    glove_path: str = "glove.6B.50d.txt",
    data_cache_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained", "imdb_data.pkl"),
    show_plot: bool = True,
):
    data_layer = get_data_layer(data_dir, glove_path, max_seq_len, vocab_size, embedding_dim, data_cache_path)

    if os.path.isfile(save_path):
        print("Loading saved model ...")
        net = NeuralNetwork.load(save_path, data_layer)
    else:
        print("Building new model ...")
        net = build(
            cell_type=cell_type,
            input_size=embedding_dim,
            hidden_size=hidden_size,
            output_size=2,
        )
        net.data_layer = data_layer

    print(f"Training for {iterations} iterations ...")
    net.train(iterations)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    NeuralNetwork.save(save_path, net)
    print("Model saved.")

    if show_plot:
        window = 1000
        moving_avg_loss = np.cumsum(net.loss, dtype=float)
        moving_avg_loss = (moving_avg_loss[window:] - moving_avg_loss[:-window]) / window
        plt.figure(f"Sentiment {cell_type.upper()} — Training Loss")
        plt.plot(moving_avg_loss, "-o", label="Moving Average Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"Sentiment{cell_type.upper()} on IMDB")
        plt.legend()
        plt.tight_layout()
        plt.show()

    test_reviews, test_labels = net.data_layer.get_test_set()
    predictions = np.array([net.test(review)[0] for review in test_reviews])
    accuracy = Helpers.calculate_accuracy(predictions, test_labels)
    print(f"\nIMDB test accuracy: {np.round(accuracy * 100, 2)}%")

    # pred_classes = np.argmax(predictions[:10], axis=1)
    # ground_truth = np.argmax(test_labels[:10], axis=1)
    # label_names = ["negative", "positive"]
    # print("\nSample predictions (first 10 test reviews):")
    # print(f"  Predicted : {[label_names[p] for p in pred_classes]}")
    # print(f"  Actual    : {[label_names[g] for g in ground_truth]}")

    return net
