from functools import reduce
import time
import matplotlib.pyplot as plt

def LU4D(n):
    # Uzupełnienie diagonali macierzy A
    matrix = []

    matrix.append([0] + [0.3] * (n - 1))  # Dolna diagonalna (pod główną)
    matrix.append([1.01] * n)  # Główna diagonalna

    matrix.append([0.2 / i for i in range(1, n)] + [0])  # Górna diagonalna (nad główną)
    matrix.append([0.15 / i ** 2 for i in range(1, n - 1)] + [0] + [0])  # Druga górna diagonalna

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


N = 11000
times = []
sizes = [i for i in range(5, N + 1,100)]

print(sizes)

for i in sizes:

    start_time = time.time()
    LU4D(i)
    end_time = time.time()

    final_time = end_time-start_time
    times.append(final_time)


plt.figure(figsize=(12, 6))  # Opcjonalne: ustawienie rozmiaru wykresu

plt.plot(sizes, times,
         marker='o',          # Kropki jako markery
         linestyle='-',       # Ciągła linia
         markersize=5,        # Rozmiar kropek
         color='#0088FE',     # Kolor niebieski
         linewidth=0)         # Grubość linii

plt.xlabel('Rozmiar danych')
plt.ylabel('Czas wykonania (sekundy)')
plt.grid(True, linestyle='--', alpha=0.7)  # Siatka z przerywaną linią

# Ustawienie marginesów, żeby punkty nie były przy krawędzi
plt.margins(x=0.02)

plt.show()


