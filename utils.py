import numexpr as ne
from sympy import symbols, sympify, latex   
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

def generate_error_graphic(error_list,equation_string,save_img=False):

    plt.plot(error_list)
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.title(equation_string)
    if save_img: plt.savefig(f'img/error.png')
    plt.grid()
    plt.show()

def generate_2d_graphic(dot_list,equation,symbols,equation_in_string,save_img=False,range_print=1):

    last_element = float(dot_list[-1][0])
    first_element = float(dot_list[0][0])

    dif = abs(last_element-first_element)

    x=np.linspace(last_element-dif, last_element+dif,100)

    plt.plot(x,[get_value( [i], equation, symbols) for i in x])
    plt.plot(dot_list,[get_value( i, equation, symbols) for i in dot_list],'xr')
    plt.grid()
    plt.title(equation_in_string)
    if save_img: plt.savefig(f'img/2d.png')
    plt.show()

def generate_3d_graphic(dot_list,equation,symbols,equation_in_string,save_img=False,range_print=1,x_label='x',y_label='y'):

    last_element_x = float(dot_list[-1][0])
    last_element_y = float(dot_list[-1][1])
    first_element_x = float(dot_list[0][0])
    first_element_y = float(dot_list[0][1])

    dif_x, dif_y = abs(last_element_x-first_element_x), abs(last_element_y-first_element_y)

    x = np.linspace(last_element_x-dif_x, last_element_x+dif_x, len(dot_list))
    y = np.linspace(last_element_y-dif_y, last_element_y+dif_y, len(dot_list))
    x, y = np.meshgrid(x, y)

    def evaluate_function(x,y):
        return ne.evaluate(equation_in_string)

    z = evaluate_function(x,y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot([i[0] for i in dot_list], [i[1] for i in dot_list], [get_value(i, equation, symbols) for i in dot_list])
    ax.plot_surface(x,y,z,cmap='coolwarm', alpha=0.6)
    #ax.contour(x, y, z, 30, cmap='viridis', linestyles='solid')  # Níveis no plano Z = -1

    ax.set_title(equation_in_string)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel('Z')
    ax.legend()
    if save_img: plt.savefig(f'img/3d.png')
    plt.show()

def generate_3d_gif(dot_list,equation,symbols,equation_in_string,range_print=1,x_label='x',y_label='y'):

    last_element_x = float(dot_list[-1][0])
    last_element_y = float(dot_list[-1][1])
    first_element_x = float(dot_list[0][0])
    first_element_y = float(dot_list[0][1])

    dif_x, dif_y = abs(last_element_x-first_element_x), abs(last_element_y-first_element_y)

    x = np.linspace(last_element_x-dif_x, last_element_x+dif_x, len(dot_list))
    y = np.linspace(last_element_y-dif_y, last_element_y+dif_y, len(dot_list))
    x, y = np.meshgrid(x, y)

    def evaluate_function(x,y):
        return ne.evaluate(equation_in_string)

    z, frames, j = evaluate_function(x,y), [], 1 

    for dot in tqdm(dot_list):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot([i[0] for i in dot_list[0:j]], [i[1] for i in dot_list[0:j]], [get_value(i, equation, symbols) for i in dot_list[0:j]])
        ax.plot(dot[0], dot[1], get_value(dot, equation, symbols),'ob')
        ax.plot_surface(x,y,z,cmap='coolwarm', alpha=0.6)
        #ax.contour(x, y, z, 30, cmap='viridis', linestyles='solid')  # Níveis no plano Z = -1

        ax.set_title(equation_in_string)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel('Z')
        ax.view_init(elev=34, azim=-39,roll=0)
        ax.legend()

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        frames.append(imageio.imread(buffer))  

        buffer.close()  
        plt.close()
        j+=1

    print('Graficos criados.')
    print('Criando animação.')
    imageio.mimsave('./gif/3d.gif', frames, duration=0.2, loop=0) 
    print('Animação criada.')

def generate_countour_line_graphic(dot_list,equation,symbols,equation_in_string,save_img=False,range_print=1,x_label='x',y_label='y',create_line_connecting_dots=False): 
    
    last_element_x = float(dot_list[-1][0])
    last_element_y = float(dot_list[-1][1])
    first_element_x = float(dot_list[0][0])
    first_element_y = float(dot_list[0][1])

    dif_x, dif_y = abs(last_element_x-first_element_x), abs(last_element_y-first_element_y)

    x = np.linspace(last_element_x-dif_x-1, last_element_x+dif_x+1, len(dot_list))
    y = np.linspace(last_element_y-dif_y-1, last_element_y+dif_y+1, len(dot_list))
    x, y = np.meshgrid(x, y)

    def evaluate_function(x,y):
        return ne.evaluate(equation_in_string)

    z = evaluate_function(x,y)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([i[0] for i in dot_list], [i[1] for i in dot_list], 'xr')
    #ax.contourf(x, y, z, 50, cmap='viridis', linestyles='solid')  
    ax.contourf(x, y, z, 50, cmap='coolwarm', linestyles='solid')  
    for i in range(len(dot_list)-1):ax.plot([dot_list[i][0],dot_list[i+1][0]],[dot_list[i][1],dot_list[i+1][1]])
    ax.set_title(equation_in_string) 
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    if save_img: plt.savefig(f'img/3d_niveis.png')
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
    imageio.mimsave('./gif/linear_regression.gif', frames, duration=0.2, loop=0) 
    print('Animação criada.')
