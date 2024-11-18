from utils import create_function, get_value, generate_error_graphic, generate_2d_graphic, generate_3d_graphic, generate_gif_linear_regression
from gradient_descent import gradient_descent
import numpy as np
import matplotlib.pyplot as plt

regression_dots = [(2,3),(3,5),(5,8),(6,11)]

equation_in_string = ''
for dot in regression_dots:
    equation_in_string += f"((x*{dot[0]}+y-{dot[1]})**2)+"
equation_in_string = equation_in_string[:-1] 

print(equation_in_string)

initial_dot, range_print, learning_rate, expected_error = [5,5], 1, 0.01, 0.0001
max_iteration, create_gif = 400, True

equation, symbols = create_function(equation_in_string)
print(symbols)

dot_list, error_list = gradient_descent(equation, symbols, learning_rate=learning_rate , initial_dot=initial_dot, expected_error=expected_error, max_iteration=max_iteration)

#generate_error_graphic(error_list,equation)

print(f"coeficient a:{dot_list[-1][0]}\ncoeficient b:{dot_list[-1][1]}")

if create_gif:
    generate_gif_linear_regression(regression_dots,dot_list)
