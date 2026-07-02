import pickle
import copy

# Neural Network Skeleton
class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.optimizer = optimizer
        self.loss = list()
        self.layers = list()
        self.data_layer = None
        self.loss_layer = None

    def __getstate__(self):
        state = self.__dict__.copy()
        # data layer exclusion
        if 'data_layer' in state:
            del state['data_layer']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # data layer to none, set later
        self.data_layer = None

    @property
    def phase(self):
        return self.layers[0].testing_phase
    @phase.setter
    def phase(self, value):
        for layer in self.layers:
            layer.testing_phase = value

    def forward(self):
        # fetch input data and save the label tensor for backward
        (tensor, self.label_tensor) = self.data_layer.next()
        # propagate through all layers
        for layer in self.layers:
            tensor = layer.forward(tensor)
        # calculate data loss
        data_loss = self.loss_layer.forward(tensor, self.label_tensor)
        # calculate regularization loss
        if self.optimizer.regularizer is not None:
            data_loss += self.optimizer.regularizer.norm(tensor)
        # return total loss
        return data_loss

    def backward(self):
        # compute starting error_tensor from loss layer
        tensor = self.loss_layer.backward(self.label_tensor)
        # traverse the layers in reverse
        for layer in reversed(self.layers):
            tensor = layer.backward(tensor)

    def append_layer(self, layer):
        # deep copy optimizer to trainable layer (don't want references to another object (have own state))
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)
        # append layer regardless of trainability
        self.layers.append(layer)

    def train(self, iterations):
        # set the layers phase correctly
        self.phase = False
        # train the network iterations
        for _ in range(iterations):
            self.loss.append(self.forward())
            self.backward()

    def test(self, input_tensor):
        # set the layers phase correctly
        self.phase = True
        # pass the input tensor through all layers
        tensor = input_tensor
        for layer in self.layers:
            tensor = layer.forward(tensor)
        return tensor
    
def save(filename, neural_network):
    # pickle dump to file
    with open(filename, 'wb') as file:
        pickle.dump(neural_network, file)

def load(filename, data_layer):
    # pickle load from file
    with open(filename, 'rb') as file:
        neural_network = pickle.load(file)
    # set data layer
    neural_network.data_layer = data_layer
    return neural_network