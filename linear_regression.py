from utils import create_function, get_value, generate_error_graphic, generate_2d_graphic, generate_3d_graphic, generate_gif_linear_regression, generate_countour_line_graphic, generate_3d_gif
from gradient_descent import gradient_descent
import numpy as np
import matplotlib.pyplot as plt
import json

with open('setting_linear_regression.json', 'r') as arquivo:
    settings = json.load(arquivo)

print(settings)
regression_dots = settings['regression_dots']

equation_in_string = ''

for dot in regression_dots:
    equation_in_string += f"((x*{dot[0]}+y-{dot[1]})**2)+"
equation_in_string = equation_in_string[:-1] 

print(equation_in_string)

equation, symbols = create_function(equation_in_string)

dot_list, error_list = gradient_descent(equation, symbols, learning_rate=settings["learning_rate"] , initial_dot=settings["initial_dot"], expected_error=settings["expected_error"], max_iteration=settings["max_iteration"])

generate_error_graphic(error_list,equation)

print(f"coeficient a:{dot_list[-1][0]}\ncoeficient b:{dot_list[-1][1]}")

if settings["create_gif"]:
    generate_gif_linear_regression(regression_dots,dot_list)
if settings["create_3d_graphic"]:
    generate_3d_graphic(dot_list,equation,symbols,equation_in_string,settings["save_img"],settings["range_print"],x_label='a',y_label='b')
    if settings["countour_line_graphic"]:
        generate_countour_line_graphic(dot_list,equation,symbols,equation_in_string,settings["save_img"],settings["range_print"],x_label='a',y_label='b')
if settings["generate_3d_gif_condition"]:
    generate_3d_gif(dot_list,equation,symbols,equation_in_string,range_print=settings["range_print"])
