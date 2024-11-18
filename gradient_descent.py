import random
from utils import create_function, get_value, product_vector
import numpy as np

def gradient_descent(equation, symbols, learning_rate, expected_error = 0.01, max_iteration = 100, stop_type = 1, initial_dot=None):

    error, dot_list, error_list = 9999999999, [], []
    current_iteration = 1

    if initial_dot is None:
        x = [random.randint(0,1000) for _ in range(len(symbols))]
    else:
        x = initial_dot
    
    while (current_iteration < max_iteration):

        print(f"\nIteration: {current_iteration}")
        print(f"x:{x}")
        
        previous_x = x.copy()

        f_x = get_value(x, equation, symbols)

        current_iteration += 1

        gradient = calculate_gradient(equation, x, symbols)

        x -= learning_rate*gradient  

        new_f_x = get_value(x, equation, symbols)   
        error = abs((new_f_x-f_x)/f_x)
        #subtraction_x = x-previous_x
        #error = abs(product_vector(subtraction_x, subtraction_x))

        print(f"Error: {error}")

        dot_list.append(x.copy())
        error_list.append(error)

        if error < expected_error:
            print('Reach out convergence')
            break

    if current_iteration >= max_iteration: print('Max Iteration')

    return dot_list, error_list   

def calculate_gradient(equation, x, symbols, delta = 0.01):

    gradient_list = []

    for index in range(len(symbols)):

        modified_x = x.copy()
        modified_x[index] += delta

        gradient_i = (get_value(modified_x, equation, symbols) - get_value(x, equation, symbols))/(delta)

        gradient_list.append(gradient_i)

    return np.array(gradient_list)
