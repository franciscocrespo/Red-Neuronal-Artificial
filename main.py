from RedNeuronal import RedNeuronal

red = RedNeuronal([2, 4, 2])

red.set_input_data([[1, 2], [2, 2]])
red.forward()
a = red.get_a()
w = red.get_weights()

print('salida {0}'.format(a))
print('pesos {0}'.format(w))
