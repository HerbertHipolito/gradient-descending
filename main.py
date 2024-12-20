from utils import create_function, get_value, generate_error_graphic, generate_2d_graphic, generate_3d_graphic, generate_countour_line_graphic, generate_3d_gif
from gradient_descent import gradient_descent
import numpy as np
import matplotlib.pyplot as plt
import json

with open('setting.json', 'r') as arquivo:
    settings = json.load(arquivo)

print(settings)
#equation_in_string = "3*x**2+2*(y-3)**2"
#equation_in_string = "-1/((x**2+y**2+1)**(1/2))+2.71**(-x**2-y**2)"
#equation_in_string  = "x**2-3*x" # one variable(x), change the initial_dot

equation, symbols = create_function(settings["equation_in_string"])

dot_list, error_list = gradient_descent(equation, symbols, learning_rate=settings["learning_rate"], initial_dot=settings["initial_dot"], expected_error=settings["expected_error"], momentum=settings["momentum"])

generate_error_graphic(error_list,equation,settings['save_img'])

if len(symbols) == 1:
    generate_2d_graphic(dot_list,equation,symbols,settings["equation_in_string"],settings["save_img"],settings["range_print"])
if len(symbols) == 2:
    generate_3d_graphic(dot_list,equation,symbols,settings["equation_in_string"],settings["save_img"],settings["range_print"])
    if settings["countour_line_graphic"]:
        generate_countour_line_graphic(dot_list,equation,symbols,settings["equation_in_string"],settings["save_img"],settings["range_print"],settings["create_line_connecting_dots"])
if settings["generate_3d_gif_condition"]:
    generate_3d_gif(dot_list,equation,symbols,settings["equation_in_string"],range_print=settings["range_print"])

