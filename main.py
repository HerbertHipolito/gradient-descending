from utils import create_function, get_value, generate_error_graphic, generate_2d_graphic, generate_3d_graphic
from gradient_descent import gradient_descent
import numpy as np
import matplotlib.pyplot as plt

equation_in_string = "3*x**2+2*(y-3)**2"
#equation_in_string  = "x**2+y**2"
#equation_in_string  = "x**2-3*x"
initial_dot, range_print = [1,1], 2 
save_img = True

equation, symbols = create_function(equation_in_string)
print(symbols)

dot_list, error_list = gradient_descent(equation, symbols, learning_rate=0.1, initial_dot=initial_dot, expected_error=0.0001)

generate_error_graphic(error_list,equation)

if len(symbols) == 1:
    generate_2d_graphic(dot_list,equation,symbols,equation_in_string,save_img,range_print)
if len(symbols) == 2:
    generate_3d_graphic(dot_list,equation,symbols,equation_in_string,save_img,range_print)