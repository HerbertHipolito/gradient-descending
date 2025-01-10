from utils import create_function, get_value, generate_error_graphic, generate_2d_graphic, generate_3d_graphic, generate_countour_line_graphic, generate_3d_gif
from gradient_descent import gradient_descent, gradient_descent_multi_exec
import numpy as np
import matplotlib.pyplot as plt
import json

with open('setting_multi_exec.json', 'r') as arquivo:
    settings = json.load(arquivo)

print(settings)
#equation_in_string = "3*x**2+2*(y-3)**2"
#equation_in_string = "-1/((x**2+y**2+1)**(1/2))+2.71**(-x**2-y**2)"
#equation_in_string  = "x**2-3*x" # one variable(x), change the initial_dot

equation, symbols = create_function(settings["equation_in_string"])

dot_dict = gradient_descent_multi_exec(equation, symbols, learning_rate=settings["learning_rate"], initial_dot=settings["initial_dot"], expected_error=settings["expected_error"], momentum=settings["momentum"],max_iteration=settings["max_iteration"])

#generate_error_graphic(error_list,equation,settings['save_img'])

#if len(symbols) == 1:
#    generate_2d_graphic(dot_list,equation,symbols,settings["equation_in_string"],settings["save_img"],settings["range_print"])
#if len(symbols) == 2:
#    generate_3d_graphic(dot_list,equation,symbols,settings["equation_in_string"],settings["save_img"],settings["range_print"],red_x_instead_of_line=settings["red_x_instead_of_line"])
#    if settings["countour_line_graphic"]:
#        generate_countour_line_graphic(dot_list,equation,symbols,settings["equation_in_string"],settings["save_img"],settings["range_print"],settings["create_line_connecting_dots"])
if settings["generate_3d_gif_condition"]:
    generate_3d_gif(dot_dict,equation,symbols,settings["equation_in_string"],range_print=settings["range_print"])

