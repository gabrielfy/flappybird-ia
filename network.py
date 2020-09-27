from keras.models import Sequential, clone_model
from keras.layers import Dense
import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_dim, hidden_neurons, output_neurons, model=None):
        """
        :param input_dim:
        :param hidden_neurons:
        :param output_neurons:
        :param model:
        """
        self.input_dim = input_dim
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons
        self.model = model
        if self.model is None:
            self._compile()

    def _compile(self):
        """
        :return:
        """
        self.model = Sequential()
        self.model.add(Dense(units=self.hidden_neurons,
                             activation='sigmoid', input_dim=self.input_dim))
        self.model.add(Dense(units=self.output_neurons, activation='softmax'))

    def save(self, path):
        """
        :param path:
        :return:
        """
        self.model.save_weights(path)

    def load(self, path):
        """
        :param path:
        :return:
        """
        self.model.load_weights(path)

    def mutation(self, lr):
        """
        :param lr:
        :return:
        """
        for layer in self.model.layers:
            weights = layer.get_weights()
            for i in range(len(weights[0])):
                for j in range(len(weights[0][i])):
                    new_weights = NeuralNetwork._update_weight(
                        weights[0][i][j], lr)
                    if new_weights is not None:
                        weights[0][i][j] = new_weights

            layer.set_weights(weights)

    @staticmethod
    def _update_weight(weight, lr):
        """
        :param weight:
        :param lr:
        :return:
        """
        if np.random.uniform(0, 1) < lr:
            return weight * np.random.normal(0, 1)
        return None

    def predict(self, inputs):
        """
        :param inputs:
        :return:
        """
        return self.model.predict(inputs)[0]

    def copy(self):
        """
        :return:
        """
        return NeuralNetwork(input_dim=self.input_dim, hidden_neurons=self.hidden_neurons, output_neurons=self.output_neurons, model=clone_model(self.model))
