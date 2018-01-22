from RedNeuronal import RedNeuronal

red = RedNeuronal([2, 4, 2])
ws = red.get_weights()
a = red.get_output()
b = red.get_bias()
print(ws)
print(a)
print(b)

red.set_input_data([1, 2])
print(a)
