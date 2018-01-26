""" 
    Este algoritmo se ha hecho para mostror como funciona el descenso de gradiente en una dimension.
    Queremos calcular la solución de la función f(x)=x**4−3x**3+2 mediante una serie de iteraciones.
    
    La derivada de la función es: f'(x) = 4x**3 - 9x**2
    Usaremos una tasa de aprendizaje de 0,01 y una precision de 0,00001
    Partiremos de un valor de x igual a 6
"""

learning_range = 0.01
precision = 0.00001
cur_x = 6
error = cur_x
df = lambda x: 4 * x**3 - 9 * x**2

i = 0
while error > precision:
    prev_x = cur_x
    cur_x = cur_x - (learning_range * df(prev_x))
    error = abs(cur_x - prev_x)
    i += 1

print("La solución, en {0} iteraciones, para la ecuación es: {1}".format(i, cur_x))