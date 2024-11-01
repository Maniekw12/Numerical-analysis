from functools import reduce
import time
import matplotlib.pyplot as plt


def LU4D(n):
    matrix = []
    matrix.append([0] + [0.3] * (n - 1))  # Lower diagonal
    matrix.append([1.01] * n)  # Main diagonal
    matrix.append([0.2 / i for i in range(1, n)] + [0])  # Upper diagonal
    matrix.append([0.15 / (i * i * i) for i in range(1, n - 1)] + [0] + [0])  # Second upper diagonal

    x = list(range(1, n + 1))


    i= 0
    while(i < n):
        if(i==0):
            matrix[0][i] = matrix

        elif(i == n-2):
            matrix[0][n - 2] = matrix[0][n - 2] / matrix[1][n - 3]
            matrix[1][n - 2] = matrix[1][n - 2] - matrix[0][n - 2] * matrix[2][n - 3]
            matrix[2][n - 2] = matrix[2][n - 2] - matrix[0][n - 2] * matrix[3][n - 3]
        elif(i == n-1):
            matrix[0][n - 1] = matrix[0][n - 1] / matrix[1][n - 2]
            matrix[1][n - 1] = matrix[1][n - 1] - matrix[0][n - 1] * matrix[2][n - 2]
        else:
            matrix[0][i] = matrix[0][i] / matrix[1][i - 1]
            matrix[1][i] = matrix[1][i] - matrix[0][i] * matrix[2][i - 1]
            matrix[2][i] = matrix[2][i] - matrix[0][i] * matrix[3][i - 1]
        i +=1

    for i in range(1, n):
        x[i] = x[i] - matrix[0][i] * x[i - 1]

    x[n - 1] = x[n - 1] / matrix[1][n - 1]
    x[n - 2] = (x[n - 2] - matrix[2][n - 2] * x[n - 1]) / matrix[1][n - 2]

    for i in range(n - 3, -1, -1):
        x[i] = (x[i] - matrix[3][i] * x[i + 2] - matrix[2][i] * x[i + 1]) / matrix[1][i]
    determinant = reduce(lambda a, b: a * b, matrix[1])

    return x, determinant


N = 300
times = []
sizes = [i for i in range(5, N + 1,1)]



wyznacznik, wynik = LU4D(300)
print("Wyznacznik")
print(wynik)
print("MWynik")
print(wyznacznik)



for i in sizes:

    start_time = time.time()
    LU4D(i)
    end_time = time.time()

    final_time = end_time-start_time
    times.append(final_time)


plt.figure(figsize=(12, 6))

plt.plot(sizes, times,
         marker='o',
         linestyle='-',
          color="b")

plt.xlabel('Rozmiar danych')
plt.ylabel('Czas wykonania (sekundy)')
plt.grid(True)
plt.show()







