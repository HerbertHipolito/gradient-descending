from sympy import symbols, sympify
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import ast
import inspect
import imageio
from io import BytesIO
from tqdm import tqdm

def create_function(expressao_str):
    variaveis = sorted({char for char in expressao_str if char.isalpha()})
    
    simbolos = symbols(variaveis)
    
    expressao = sympify(expressao_str)
    
    return expressao, simbolos

def get_value(values, equation, symbols):

    if len(values) != len(symbols):
        print(values)
        print(symbols)
        raise Exception("Size of values and symbols are different")

    values_symbols = {symbols[index]:values[index] for index in range(len(symbols))}

    return equation.evalf(subs=values_symbols)

def generate_error_graphic(error_list,equation_string):

    plt.plot(error_list)
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.title(equation_string)
    plt.grid()
    plt.show()

def generate_2d_graphic(dot_list,equation,symbols,equation_in_string,save_img=False,range_print=1):

    last_element = float(dot_list[-1][0])
    x=np.linspace(last_element-range_print, last_element+range_print,100)

    plt.plot(x,[get_value( [i], equation, symbols) for i in x])
    plt.plot(dot_list,[get_value( i, equation, symbols) for i in dot_list],'xr')
    plt.title(equation_in_string)
    if save_img: plt.savefig(f'img/2d.png')
    plt.show()

def generate_3d_graphic(dot_list,equation,symbols,equation_in_string,save_img=False,range_print=1):

    last_element_x = float(dot_list[-1][0])
    last_element_y = float(dot_list[-1][1])

    x = np.linspace(last_element_x-range_print, last_element_x+range_print, len(dot_list))
    y = np.linspace(last_element_y-range_print, last_element_y+range_print, len(dot_list))
    x, y = np.meshgrid(x, y)

    #z = x**2 + y**2
    def my_expression(x,y):
        return 3*x**2+2*(y-3)**2

    z = my_expression(x,y)

    z_string = to_string(my_expression)

    if z_string.split('return ')[-1] != equation_in_string: raise Exception('Equation in generate_3d_graphic not adjusted, change the equation in my_expression function ')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot([i[0] for i in dot_list], [i[1] for i in dot_list], [get_value(i, equation, symbols) for i in dot_list], 'xr')
    ax.plot_surface(x,y,z,cmap='coolwarm', alpha=0.6)
    ax.set_title(equation_in_string)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    if save_img: plt.savefig(f'img/3d.png')

    plt.show()

def product_vector(v1,v2):

    sum = 0
    for i in range(len(v1)):
        sum += v1[i]*v2[i]
    
    return sum

def to_string(expression):
    return inspect.getsource(expression).strip()

def regression(x,a,b):
    return a * x + b

def linear_regression(regression_dots):
    
    x = np.array([dot[0] for dot in regression_dots])
    y = np.array([dot[1] for dot in regression_dots])

    n = len(x)
    a = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
    b = (np.sum(y) - a * np.sum(x)) / n

    x_reta = np.linspace(min(x), max(x), 100)
    
    y_reta = regression(x_reta,a,b)

    return x_reta, y_reta, x, y, a, b

def generate_gif_linear_regression(regression_dots,dot_list):
    
    frames = []
    x_reta, y_reta, x, y, a, b = linear_regression(regression_dots)
    i  = 0

    print('Criando graficos...')
    for dot in tqdm(dot_list):
        
        a_gradient = dot[0]
        b_gradient = dot[1]
        y_reta_gradient = regression(x_reta,a_gradient,b_gradient)

        plt.figure(figsize=(8, 6))
        plt.grid()
        plt.scatter(x, y, color='blue', label='Pontos Dados')
        plt.plot(x_reta, y_reta, color='red', label=f'Reta de Regressão: y = {a:.2f}x + {b:.2f}')
        plt.plot(x_reta, y_reta_gradient, color='green', label=f' Aproximação da Reta de Regressão: y = {a_gradient:.2f}x + {b_gradient:.2f}') 
        plt.title('Reta de Regressão')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        filename = f"frame_{i}.png"
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        frames.append(imageio.imread(buffer))  
        buffer.close()  
        plt.close()
        i+=1

    print('Graficos criados.')
    print('Criando animação.')
    imageio.mimsave('linear_regression.gif', frames, duration=0.2, loop=0) 
    print('Animação criada.')
