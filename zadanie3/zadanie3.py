from functools import reduce
import numpy as np
import time


def LU4D(n):
    # Uzupełnienie diagonali macierzy A
    matrix = []
    matrix.append([0] + [0.2] * (n - 1))  # Dolna diagonalna (pod główną)
    matrix.append([1.2] * n)  # Główna diagonalna
    matrix.append([0.1 / i for i in range(1, n)] + [0])  # Górna diagonalna (nad główną)
    matrix.append([0.4 / i ** 2 for i in range(1, n - 1)] + [0] + [0])  # Druga górna diagonalna

    # Stworzenie wektora wyrazów wolnych
    x = list(range(1, n + 1))

    # Rozkład LU
    for i in range(1, n - 2):
        matrix[0][i] = matrix[0][i] / matrix[1][i - 1]
        matrix[1][i] = matrix[1][i] - matrix[0][i] * matrix[2][i - 1]
        matrix[2][i] = matrix[2][i] - matrix[0][i] * matrix[3][i - 1]

    matrix[0][n - 2] = matrix[0][n - 2] / matrix[1][n - 3]
    matrix[1][n - 2] = matrix[1][n - 2] - matrix[0][n - 2] * matrix[2][n - 3]
    matrix[2][n - 2] = matrix[2][n - 2] - matrix[0][n - 2] * matrix[3][n - 3]

    matrix[0][n - 1] = matrix[0][n - 1] / matrix[1][n - 2]
    matrix[1][n - 1] = matrix[1][n - 1] - matrix[0][n - 1] * matrix[2][n - 2]

    # Podstawianie w przód
    for i in range(1, n):
        x[i] = x[i] - matrix[0][i] * x[i - 1]

    # Podstawiania w tył
    x[n - 1] = x[n - 1] / matrix[1][n - 1]
    x[n - 2] = (x[n - 2] - matrix[2][n - 2] * x[n - 1]) / matrix[1][n - 2]

    for i in range(n - 3, -1, -1):
        x[i] = (x[i] - matrix[3][i] * x[i + 2] - matrix[2][i] * x[i + 1]) / matrix[1][i]

    # Obliczanie wartości wyznacznika macierzy
    wyznacznik = reduce(lambda a, b: a * b, matrix[1])

    return x, wyznacznik



n= 300
#dodajemy pierwsza poddiabgonale o n elementach
#mamy teraz macierz 4x4 gdzie
#[0] - pierwsza diagonala
#[1] - diagonala
#[2] - trzecia diaboanala
#[3] - czwarta daigonala


x, determinant = LU4D(n)
print("Szukane rozwiązanie to: ", x)
print()
print("Wyznacznik macierzy A = ", determinant)
