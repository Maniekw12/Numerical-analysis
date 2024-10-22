import numpy as np
import matplotlib.pyplot as plt

""" sin(x^3) funkcja"""
def f1(x):
    return np.sin(x**3)

def f1_derivate(x):
    return 3 * x ** 2 * np.cos(x ** 3)

""" cos(x^3) funkcja """
def f2(x):
    return np.cos(x**3)

def f2_derivate(x):
    return -3 * x ** 2 * np.sin(x ** 3)

""" wzory na pochodną - centraln"""
def derivate_func_a(f, x, h):
    return (f(x + h) - f(x)) / h

def derivate_func_b(f, x, h):
    return (f(x+h) - f(x-h))/(2*h)

"""Funkcja do analizowania bledow"""
def analyze_errors(x, h_values, function, function_derivate):
    errors_in_function_a = []
    errors_in_function_b = []

    for h in h_values:
        """Przyblizone wartosci pochodnych na podstawie 
            funkcji a, b"""
        approx_in_a = derivate_func_a(function, x, h)
        approx_in_b = derivate_func_b(function, x, h)

        """Bledy w a oraz bledy w b"""
        error_in_a = np.abs(approx_in_a - function_derivate(x))
        error_in_b = np.abs(approx_in_b - function_derivate(x))

        errors_in_function_a.append(error_in_a)
        errors_in_function_b.append(error_in_b)

    return errors_in_function_a, errors_in_function_b

#############main##################
x = 0.2
"""Bledy dla Double - sin(x^3)"""
h_values_Double64 = np.logspace(-18, 0, 100, dtype=np.float64)
errors_a_64, errors_b_64 = analyze_errors(x, h_values_Double64, f1, f1_derivate)

"""Bledy dla Float - sin(x^3)"""
h_values_float32 = list(h_values_Double64.astype(np.float32))
errors_a_32, errors_b_32 = analyze_errors(np.float32(x), h_values_float32, f1, f1_derivate)

"""Bledy dla Double - cos(x^3)"""
h_values_Double64_2 = np.logspace(-18, 0, 100, dtype=np.float64)
errors_a_64_2, errors_b_64_2 = analyze_errors(x, h_values_Double64_2, f2, f2_derivate)

"""Bledy dla Float - cos(x^3)"""
h_values_float32_2 = list(h_values_Double64_2.astype(np.float32))
errors_a_32_2, errors_b_32_2 = analyze_errors(np.float32(x), h_values_float32_2, f2, f2_derivate)

"""Rysowanie wykresu"""
plt.figure(figsize=(12, 12))

plt.subplot(2, 1, 1)

plt.loglog(h_values_Double64, errors_a_64, 'b-', label="(A) Double(64) sin(x^3)", markersize=4)
plt.loglog(h_values_Double64, errors_b_64, 'r-', label="(B) Double(64) sin(x^3)", markersize=4)
plt.loglog(h_values_float32, errors_a_32, 'g--', label="(A) Float(32) sin(x^3)", markersize=4)
plt.loglog(h_values_float32, errors_b_32, 'm--', label="(B) Float(32) sin(x^3)", markersize=4)

plt.xlabel('h')
plt.ylabel('E(h) |Dh f(x) - f\'(x)|')
plt.title('Error Analysis for Derivative Approximations: sin(x^3)')
plt.legend()
plt.grid(True)

"""wykres dla 2 funkcji"""
plt.subplot(2, 1, 2)

plt.loglog(h_values_Double64_2, errors_a_64_2, 'b-', label="(A) Double(64) cos(x^3)", markersize=4)
plt.loglog(h_values_Double64_2, errors_b_64_2, 'r-', label="(B) Double(64) cos(x^3)", markersize=4)
plt.loglog(h_values_float32_2, errors_a_32_2, 'g--', label="(A) Float(32) cos(x^3)", markersize=4)
plt.loglog(h_values_float32_2, errors_b_32_2, 'm--', label="(B) Float(32) cos(x^3)", markersize=4)

plt.xlabel('h')
plt.ylabel('E(h) |Dh f(x) - f\'(x)|')
plt.title('Error Analysis for Derivative Approximations: cos(x^3)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
