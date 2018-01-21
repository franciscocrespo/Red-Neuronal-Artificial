import random, math, os


class RedNeuronal:
    """
    This class received an array. The dimension of array is equal to number of layers and this contain
    the number of neurons by layer.
    """
    def __init__(self, redArray, input = None):
        self.__k = len(redArray)
        self.__neurons_for_layer = [i for i in redArray]
        self.__a = [0 for i in range(self.__neurons_for_layer[self.__k - 1])]
        self.__weights = [[[random.uniform(0.1, 0.2)
                            for j in range(self.__neurons_for_layer[k + 1])]
                           for i in range(self.__neurons_for_layer[k])]
                          for k in range(self.__k - 1)]
        self.__b = [[random.uniform(0.1, 0.2) for i in range(k)] for k in self.__neurons_for_layer]
        self.__input = input

    # This method gets the weights
    def get_weights(self):
        return self.__weights

    # This method gets the output of the last layer
    def get_output(self):
        return self.__a

    # This method gets the bias input
    def get_bias(self):
        return self.__b

    # This method applies the sigma function to the input of a neuron
    def __sigmoid_function(self, ai):
        return 1 / (1 + math.exp(-ai))

    # This method gets the output i in the layer k of a neuron
    def __ai_calculate(self, idx, layer_number):
        ak_i = 0
        if layer_number == 0:
            ak_i = self.__input[idx]
        if layer_number > 0:
            bi = self.__bias[layer_number][idx]
            sm = sum([self.__a[layer_number - 1][j] * self.__weights[layer_number - 1][j][idx]
                       for j in range(layer_number - 1)])
            net = bi + sm
            ak_i = self.__sigmoid_function(net)
        return ak_i

    # This method gets the global outputs of the artificial neural network
    def forward(self):
        pass