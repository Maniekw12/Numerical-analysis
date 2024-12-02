import matplotlib.pyplot as plt
import time
import numpy as np

def check(n):
    A = np.diag([4] * n) + np.diag([2] * (n - 1), 1) + np.ones((n, n))
    y_numpy = np.linalg.solve(A, b)
    return y_numpy


def countTime1(times,N_values,N):
    for N in N_values:
        start_time = time.time()
        matrix = [[4] * N, [2] * (N - 1) + [0]]
        b = [2] * N
        sherman_morrison(matrix, b, N)
        times.append((time.time() - start_time))


def countTime2(times1,N_values1,N):
    for N in N_values1:
        start_time = time.time()
        A = np.diag([4] * N) + np.diag([2] * (N - 1), 1) + np.ones((N, N))
        b = [2] * N
        y_numpy = np.linalg.solve(A, b)
        times1.append((time.time() - start_time))


def sherman_morrison(matrix, b, n):
    y = []
    q = [0] * n
    w = [0] * n

    q[n - 1] = b[n - 1] / matrix[0][n - 1]
    w[n - 1] = 1 / matrix[0][n - 1]

    for i in range(n - 2, -1, -1):
        q[i] = (b[i] - matrix[1][i] * q[i + 1]) / matrix[0][i]
        w[i] = (1 - matrix[1][i] * w[i + 1]) / matrix[0][i]


    delta = sum(q) / (1 + sum(w))

    for i in range(n):
        y.append(q[i] - w[i] * delta)

    return y

n = 120
matrix = []
matrix.append([4] * n)
matrix.append([2] * (n - 1) + [0])
b = [2] * n

y_custom = sherman_morrison(matrix, b, n)
print("Sherman-morrison:")
print(y_custom)

y_numpy = check(n)

print("numpy:")
print(y_numpy)



N_values = range(10, 10000, 1000)
times = []
countTime1(times,N_values,n)

N_values1 = range(10, 10000, 500)
times1 = []
countTime2(times1,N_values1,n)



plt.plot(N_values, times,label="Algorytm wykorzystujący budowę macierzy")
plt.plot(N_values1, times1,label="Numpy")
plt.grid(True)
plt.xlabel("Rozmiar macierzy N")
plt.ylabel("Czas rozwiązania [s]")
plt.title("Czas rozwiązania w funkcji rozmiaru N")
plt.legend()
plt.show()