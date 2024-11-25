import numpy as np
import matplotlib.pyplot as plt


def create_matrix_and_vector(N, d):

    A = np.zeros((N, N))
    b = np.arange(1, N + 1)

    for i in range(N):
        A[i, i] = d
        if i > 0:
            A[i, i - 1] = 0.5
        if i < N - 1:
            A[i, i + 1] = 0.5
        if i > 1:
            A[i, i - 2] = 0.1
        if i < N - 2:
            A[i, i + 2] = 0.1

    return A, b

# w mojej implementacji kod wykonuje sie do momentu, az jest wystarczajaco bliski rozwiazania albo dokona sie maksymalna liczba iteracji

def jacobi_method(A, b, x_start=None, tol=1e-12, max_iterations=100):

    N = len(b)
    x = np.zeros(N) if x_start is None else np.array(x_start)
    x_new = np.zeros(N)
    iterations = []
    errors = []

    for k in range(max_iterations):
        for i in range(N):
            x_new[i] = (b[i] - np.sum(A[i, :i] * x[:i]) - np.sum(A[i, i + 1:] * x[i + 1:])) / A[i, i]

        # Obliczanie błędu
        error = np.linalg.norm(x_new - x, ord=np.inf)
        errors.append(error)
        iterations.append(k + 1)

        if error < tol:
            break

        x[:] = x_new

    return x, iterations, errors


def gauss_seidel_method(A, b, x_start=None, tol=1e-12, max_iterations=1000):

    N = len(b)
    x = np.zeros(N) if x_start is None else np.array(x_start)
    iterations = []
    errors = []

    for k in range(max_iterations):
        x_old = x.copy()

        for i in range(N):
            x[i] = (b[i] - np.sum(A[i, :i] * x[:i]) - np.sum(A[i, i + 1:] * x[i + 1:])) / A[i, i]

        # Obliczanie błędu
        error = np.linalg.norm(x - x_old, ord=np.inf)
        errors.append(error)
        iterations.append(k + 1)

        if error < tol:
            break

    return x, iterations, errors




N = 200  
d_values = [2.0,5.0, 10.0]
start_points = [
    np.zeros(N),  # Punkt startowy: zerowy wektor
    np.ones(N) * 60,   # Punkt startowy: wektor z samymi 60tkami
    np.random.rand(N)* 4  # Punkt startowy: losowy wektor
]

for d in d_values:

    A, b = create_matrix_and_vector(N, d)

    x_exact = np.linalg.solve(A, b)

    for idx, x_start in enumerate(start_points):
        x_jacobi, it_jacobi, err_jacobi = jacobi_method(A, b, x_start=x_start)

        x_gs, it_gs, err_gs = gauss_seidel_method(A, b, x_start=x_start)

        plt.figure(figsize=(10, 6))
        plt.semilogy(it_jacobi, err_jacobi, label='Jacobi')
        plt.semilogy(it_gs, err_gs, label='Gauss-Seidel')
        plt.xlabel('Liczba iteracji')
        plt.ylabel('Błąd')
        plt.title(f'Zbieżność metod iteracyjnych (d = {d}, punkt startowy #{idx + 1})')
        plt.legend()
        plt.grid()
        plt.show()

        print(f"=== d = {d}, Punkt startowy #{idx + 1} ===")
        print(f"Błąd końcowy (Jacobi): {np.linalg.norm(x_exact - x_jacobi):.2e}")
        print(f"Błąd końcowy (Gauss-Seidel): {np.linalg.norm(x_exact - x_gs):.2e}")
        print()
