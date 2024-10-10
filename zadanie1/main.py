import numpy as np
import matplotlib.pyplot as plt



def f(x):
    return np.sin(x**3)

def f_derivate(x):
    return 3 * x ** 2 * np.cos(x ** 3)

def func_a(f,x,h):
    return (f(x + h) - f(x)) / h

def func_b(f,x,h):
    return (f(x+h) - f(x-h))/(2*h)

"""Funkcja do analizowania bledow"""
"""Robimy 2 funkcje w ktorych sprawdzamy"""
def analyze_errors(x,h_values,function, function_derivate):
    errors_in_function_a = []
    errors_in_function_b = []


    for h in h_values:
        """Przyblizone wartosci pochodnych na podstawie 
            funkcji a, b"""
        approx_in_a = func_a(function,x,h)
        approx_in_b = func_b(function,x,h)

        """Bledy w a oraz bledy w b"""

        error_in_a = np.abs(approx_in_a - function_derivate(x))
        error_in_b = np.abs(approx_in_b - function_derivate(x))

        errors_in_function_a.append(error_in_a)
        errors_in_function_b.append(error_in_b)

    return errors_in_function_a,errors_in_function_b


#############main##################
x = 10
x = 0.2



"""Bledy dla Double"""
h_values_Double64 = np.logspace(-18, 0, 100,dtype=np.float64)
errors_a_64, errors_b_64 = analyze_errors(x,h_values_Double64,f,f_derivate)

"""Bledy dla Float"""
h_values_float32 = list(h_values_Double64.astype(np.float32))
errors_a_32, errors_b_32 = analyze_errors(x,h_values_float32,f,f_derivate)

"""Rysowanie wykresu"""
plt.figure(figsize=(10, 6))

plt.loglog(h_values_Double64, errors_a_64, 'b-', label="(A) Double(64)", markersize=4)
plt.loglog(h_values_Double64, errors_b_64, 'r-', label="(B) Double(64)", markersize=4)
plt.loglog(h_values_float32, errors_a_32, 'g--', label="(A) Float(32)", markersize=4)
plt.loglog(h_values_float32, errors_b_32, 'm--', label="(B) Float(32)", markersize=4)

plt.xlabel('h')
plt.ylabel('E(h) |Dh f(x) - f\'(x)|')
plt.title('Error Analysis for Derivative Approximations')
plt.legend()
plt.grid(True)
plt.show()
