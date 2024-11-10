import numpy as np                  # Tylko do testów
import matplotlib.pyplot as plt     # Biblioteka do tworzenia wykresow
import time
def sherman(matrix,b):


    # Backward subtitution dla obu równań
    z = [0]*n
    x = [0]*n
    z[n-1] = b[n-1] / matrix[0][n-1]

    for i in range(n - 2, -1, -1):
        z[i] = (b[n-2] - matrix[1][i] * z[i+1]) / matrix[0][i]

    x[n-1] = 1 / matrix[0][n-1]
    for i in range(n - 2, -1, -1):
        x[i] = (1 - matrix[1][i] * x[i+1]) / matrix[0][i]


    delta = sum(z)/(1+sum(x))

    # Wyliczenie wyniku
    y=[]
    for i in range(len(z)):
        y.append(z[i]-x[i]*delta)

    return y

n = 120

matrix = []
matrix.append([4] * n)
matrix.append([2] * (n-1) + [0])
b = [2] * n

print(sherman(matrix,b))

