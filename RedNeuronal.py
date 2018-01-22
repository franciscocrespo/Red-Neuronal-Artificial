import random, math, os


class RedNeuronal:
    """
    This class received an array. The dimension of array is equal to number of layers and this contain
    the number of neurons by layer.
    """

    def __init__(self, red_array):
        self.__k = len(red_array)
        self.__neurons_for_layer = [i for i in red_array]
        self.__a = [[0 for i in range(k)] for k in self.__neurons_for_layer]
        self.__weights = [[[random.uniform(0.1, 0.2)
                            for j in range(self.__neurons_for_layer[k + 1])]
                           for i in range(self.__neurons_for_layer[k])]
                          for k in range(self.__k - 1)]
        self.__b = [[random.uniform(0.1, 0.2) for i in range(k)] for k in self.__neurons_for_layer]
        self.__target = []

    # This method sets the input data to the inputs of the neural network and right outputs for each inputs
    def set_input_data(self, input_data):
        inp = input_data[0]
        target = input_data[1]
        if len(self.__a[0]) == len(inp):
            self.__a[0] = inp
        if len(self.__a[self.__k - 1]) == len(target):
            self.__target = target

    # This method gets the weights
    def get_weights(self):
        return self.__weights

    def get_weight_by_id(self, i, j, k):
        return self.__weights[k][i][j]

    # This method gets the output of the last layer
    def get_a(self):
        return self.__a

    def get_a_by_id(self, i, k):
        return self.__a[k][i]

    # This method gets the bias input
    def get_bias(self):
        return self.__b

    def get_b_by_id(self, i, k):
        return self.__b[k][i]

    # This method gets the target outputs
    def get_target(self):
        return self.__target

    def get_target_by_id(self, i):
        return self.__target[i]

    # This method gets the global outputs of the artificial neural network
    def forward(self):
        k = 1
        while k < self.__k:
            for i in range(self.__neurons_for_layer[k]):
                ak_i = self.__ai_calculate(i, k)
                self.__set_aki(ak_i, k, i)
            k += 1


    # Auxiliary functions

    # This method applies the sigma function to the input of a neuron
    @staticmethod
    def __sigmoid_function(ai):
        return 1 / (1 + math.exp(-ai))

    def __a_multiply_weight(self, k, j, i):
        return self.__a[k - 1][j] * self.__weights[k - 1][j][i]

    # This method gets the output i in the layer k of a neuron
    def __ai_calculate(self, idx, layer_number):
        ak_i = []
        if layer_number > 0 & layer_number <= self.__k:
            bi = self.__b[layer_number][idx]
            sm = sum([self.__a_multiply_weight(layer_number, j, idx) for j in
                      range(layer_number - 1)])
            net = bi + sm
            ak_i = self.__sigmoid_function(net)

        return ak_i

    def __set_aki(self, ak_i, k, i):
        self.__a[k][i] = ak_i
