import random, math, os


class RedNeuronal:
    """
    This class received an array. The dimension of array is equal to number of layers and this contain
    the number of neurons by layer.
    """

    # CONSTRUCTOR

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
        self.__errors = []

    # ----------------------------------------------------------------------------------------------------- #

    # GETTERS AND SETTERS

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
        try:
            return self.__weights[k][i][j]
        except IndexError:
            pass

    # This method gets the output of the last layer
    def get_a(self):
        return self.__a

    def get_a_by_id(self, i, k):
        try:
            return self.__a[k][i]
        except IndexError:
            pass

    # This method gets the bias input
    def get_b(self):
        return self.__b

    def get_b_by_id(self, i, k):
        try:
            return self.__b[k][i]
        except IndexError:
            pass

    # This method gets the target outputs
    def get_target(self):
        return self.__target

    def get_target_by_id(self, i):
        try:
            return self.__target[i]
        except IndexError:
            pass

    # This method gets the target outputs
    def get_errors(self):
        return self.__errors

    def get_error_by_id(self, i):
        try:
            return self.__errors[i]
        except IndexError:
            pass

    # ----------------------------------------------------------------------------------------------------- #

    # FORWARD

    # This method gets the global outputs of the artificial neural network
    def forward(self):
        k = 1
        while k < self.__k:
            neurons_for_layer = self.__neurons_for_layer[k]
            neurons_for_layer_before = self.__neurons_for_layer[k - 1]
            for i in range(neurons_for_layer):
                ak_i = self.__ai_calculate(i, k, neurons_for_layer_before)
                self.__set_aki(ak_i, k, i)
            k += 1

    # ----------------------------------------------------------------------------------------------------- #

    # BACK PROPAGATION

    # This method gets the error for the global output
    def total_error(self):
        neurons_last_k = self.__neurons_for_layer[self.__k - 1]
        for i in range(neurons_last_k):
            diff = self.__target[i] - self.__a[neurons_last_k][i]
            error = 1/2 * pow(diff, 2)
            self.__errors.append(error)

        return sum(self.__errors)

    # This Method gets the delta of the error with respect Y sub i
    def delta_error_yi(self, idx):
        try:
            last_layer = self.__last_layer()
            y_i = self.__a[last_layer][idx]
            s_i = self.__target[idx]
            de = -(s_i - y_i)
            return de
        except IndexError:
            pass

    def delta_last_layer_sub_i(self, idx):
        try:
            last_layer = self.__last_layer()
            y_i = self.__a[last_layer][idx]
            return y_i * (1 - y_i)
        except IndexError:
            pass

    def delta_hidden_layers_sub_i(self, idx, k):
        pass

    # ----------------------------------------------------------------------------------------------------- #

    # AUXILIARY FUNCTIONS

    # This method applies the sigma function to the input of a neuron
    @staticmethod
    def __sigmoid_function(ai):
        return 1 / (1 + math.exp(-ai))

    def __a_multiply_weight(self, k, j, i):
        return self.__a[k - 1][j] * self.__weights[k - 1][j][i]

    # This method gets the output i in the layer k of a neuron
    def __ai_calculate(self, idx, layer_number, neurons_for_layer_before):
        bi = self.__b[layer_number][idx]
        sm = sum([self.__a_multiply_weight(layer_number, j, idx) for j in
                      range(neurons_for_layer_before)])
        net = bi + sm
        return self.__sigmoid_function(net)

    def __set_aki(self, ak_i, k, i):
        self.__a[k][i] = ak_i

    def __last_layer(self):
        return self.__k - 1

