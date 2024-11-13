from functools import reduce
import time
import matplotlib.pyplot as plt


def LU4D(n):
    matrix = []
    matrix.append([0] + [0.3] * (n - 1))
    matrix.append([1.01] * n)
    matrix.append([0.2 / i for i in range(1, n)] + [0])
    matrix.append([0.15 / (i * i * i) for i in range(1, n - 1)] + [0] + [0])

    x = list(range(1, n + 1))


    i= 1
    while(i < n):
        if(i == n-2):
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
print(wyznacznik)
print("Wynik")
print(wynik)



for i in sizes:

    start_time = time.perf_counter()
    LU4D(i)
    end_time = time.perf_counter()

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







