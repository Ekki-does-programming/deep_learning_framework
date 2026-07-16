# NumPy deep learning framework

A deep learning framework built from scratch in `NumPy`, featuring custom implementations of core layers (Conv, Pool, FC, etc.), activation functions, and optimizers (SGD, Adam, etc.); validated by training LeNet and LSTM architecture end-to-end.
The LeNet written from scratch achieves accuracies over 97% on MNIST.

## 📂 Structure of repository

This repository is divided into the following key directories and files:
```bash
├── IMDBreviews           # Sentiment analysis using RNN/LSTM
├── Layers                # Layers for neural network construction
├── MNIST                 # Recognition analysis using LeNet
├── Models                # Build models from layers for tasks
├── NeuralNetwork.py      # General neural network class
├── Optimization          # Determine loss and improve parameters
└── requirements.txt      # Virtual environment setup
```

The main part of this numpy framework are the deep learning components which are built from scratch. These located in the [Layers](Layers), [Optimization](Optimization) and [NeuralNetwork.py](NeuralNetwork.py) folders and files. These form a framework to build custom models from.

## ⚡ Toy applications

Using this framework two toy applications are built. First, a LeNet model is constructed for hand-written digit classification using the MNIST dataset. Second, recurrent networks using either a RNN or LSTM cell are used to classify the sentiment of IMDB reviews.

### ⭐ Digit recognition

The heart of this toy application is the [LeNet model](Models/LeNet.py) which utilizes the self-build layers. It is trained on the [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database), located in `MNIST/Data`, via:
```bash
cd /path/to/project/MNIST
mkdir trained
python TrainLeNet.py
```

Already training the model on $300 \times 50$ images, leads to an accuracy of $\approx 93$%. With longer training accuracies over $97$% can be achieved.
The trained model is stored in the `trained` folder. Training can just be continued by executing the python script again.

### ⭐ Sentiment classification

The [sentiment models](Models/SentimentModel.py) either use a single RNN- or LSTM-cell. They are trained on the [large movie review dataset](https://ai.stanford.edu/~amaas/data/sentiment/) utilizing a [GloVe](https://nlp.stanford.edu/projects/glove/) embedding. This data can be downloaded executing:
```bash
cd /path/to/project/IMDBreviews
python SetupData.py
```
The models can be trained via:
```bash
mkdir trained
python TrainSentimentRNN.py # or: python TrainSentimentLSTM.py
```
The RNN and/or LSTM models will be saved in the `trained` folder. They can be tested by executing:
```bash
python PredictSentiment.py --model lstm # or: --model rnn; default is lstm
```
An interaction could look like this:
```bash
Loading cached IMDB data ...
Loading model from 'trained/SentimentLSTM' ...

1) Classify a random test review
2) Type your own review
3) Quit
> 1

Review:
I usually much prefer French movies over American ones, with explosions and car chases, but this movie was very disappointing. There is no way to write a spoiler because nothing really happens. This French couple has been living in Lisbon for years, and they return to Paris for a friend's wedding. They announce to another friend they are having dinner with that they are going to split. Then nothing much happens, they don't seem to know whether they want to separate or not. I don't necessarily th ...

Actual   : negative
Predicted: negative  (confidence 72.7%)

1) Classify a random test review
2) Type your own review
3) Quit
> 2

Type a review: It's eye-filling, well-cast, often very funny and executed with great imagination and flair.
Predicted: positive  (confidence 95.3%)

1) Classify a random test review
2) Type your own review
3) Quit
> 3
```

## ⚙️ Setup the virtual environment

A virtual environment tailored for this project can be setup via:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```
It can be deactivated typing `deactivate` into the terminal.

---

**Note:**
Originally this framework was developed as a university project. Now parts of that project are released here.

---