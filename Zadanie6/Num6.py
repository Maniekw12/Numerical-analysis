import numpy as np
import matplotlib.pyplot as plt


class Solution:
    TOLERANCE = 1e-12
    MAX_ITERATION = 1000

    def __init__(self, matrix):
        self.matrix = np.array(matrix)

    def power_method_iteration(self):
        eigenvalues = []
        errors = []

        def mat_vec_mul(matrix, vector):
            return [sum(row[i] * vector[i] for i in range(len(vector))) for row in matrix]

        def vec_norm(vector):
            return sum(xi ** 2 for xi in vector) ** 0.5

        def vec_dot(v1, v2):
            return sum(v1[i] * v2[i] for i in range(len(v1)))

        n = len(self.matrix)
        x = [1 for _ in range(n)]
        norm = vec_norm(x)
        x = [xi / norm for xi in x]

        prev_lambda = 0
        for _ in range(Solution.MAX_ITERATION):
            x_next = mat_vec_mul(self.matrix, x)
            norm = vec_norm(x_next)
            x_next = [xi / norm for xi in x_next]
            current_lambda = vec_dot(x_next, mat_vec_mul(self.matrix, x_next))
            eigenvalues.append(current_lambda)
            errors.append(abs(current_lambda - prev_lambda))

            if errors[-1] < Solution.TOLERANCE:
                break

            x = x_next
            prev_lambda = current_lambda

        return errors, eigenvalues, x

    def qr_method(self):
        A = self.matrix.copy()
        iteration = 1

        diagonal_elements = [[] for _ in range(len(A))]
        subdiag_elements = [[] for _ in range(len(A) - 1)]
        errors_qr = [[] for _ in range(len(A))]

        exact_eigenvalues, _ = np.linalg.eig(self.matrix)

        while True:
            Q, R = np.linalg.qr(A)
            A = np.dot(R, Q)

            for i in range(len(A)):
                diagonal_elements[i].append(A[i][i])
                errors_qr[i].append(abs(A[i][i] - exact_eigenvalues[i]))

            for i in range(len(A) - 1):
                subdiag_elements[i].append(abs(A[i + 1][i]))

            max_subdiag = max(abs(A[i + 1][i]) for i in range(len(A) - 1))
            if max_subdiag < Solution.TOLERANCE or iteration > Solution.MAX_ITERATION:
                break

            iteration += 1

        return A, diagonal_elements, subdiag_elements, errors_qr

    def generate_power_method_plots(self):
        errors, eigenvalues, eigenvector = self.power_method_iteration()

        exact_eigenvalues, exact_eigenvectors = np.linalg.eig(self.matrix)
        max_eigenvalue_index = np.argmax(exact_eigenvalues)
        numpy_eigenvector = exact_eigenvectors[:, max_eigenvalue_index]
        numpy_eigenvector = numpy_eigenvector / np.linalg.norm(numpy_eigenvector)

        plt.figure(figsize=(8, 6))
        plt.semilogy(errors, marker='o', label="Błąd")
        plt.title("Metoda Potęgowa")
        plt.xlabel("Iteracja")
        plt.ylabel("Błąd")
        plt.legend()
        plt.grid(True)
        plt.show()
        print("Największa wartość własna (Metoda Potęgowa):", eigenvalues[-1])
        print("Odpowiadający wektor własny (Metoda Potęgowa):", [float(_) for _ in eigenvector])
        print("Wektor własny (NumPy):", [float(_) for _ in numpy_eigenvector])

    def generate_qr_plots(self):
        A, diagonal_elements, subdiagonal_elements, errors_qr = self.qr_method()

        plt.figure(figsize=(11, 7))
        labels = ['1 element', '2 element', '3 element', '4 element']
        for i, error in enumerate(errors_qr):
            plt.semilogy(range(1, len(error) + 1), error, label=labels[i], linewidth=2)
        plt.xlabel("Iteracja", fontsize=16)
        plt.title("Różnice między elementami diagonalnymi a dokładnymi wartościami własnymi", fontsize=16)
        plt.ylabel("Różnica", fontsize=16)
        plt.legend(fontsize=10, loc='upper right')
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(11, 7))
        labels = ['1-szy element poddiagonalny', '2-gi element poddiagonalny', '3-ci element poddiagonalny']
        for i, subdiag in enumerate(subdiagonal_elements):
            plt.semilogy(range(1, len(subdiag) + 1), subdiag, label=labels[i], linewidth=2)
        plt.xlabel("Iteracja", fontsize=16)
        plt.title("Wartości elementów poddiagonalnych w k-tym kroku", fontsize=18)
        plt.ylabel("Wartość", fontsize=16)
        plt.legend(fontsize=10, loc='upper right')
        plt.grid(True)
        plt.show()

        print("Wartości własne (Metoda QR):", [float(diag[-1]) for diag in diagonal_elements])

    def run_all(self):
        print("-" * 30 + "Metoda Potęgowa" + "-" * 30)
        self.generate_power_method_plots()

        print("-" * 32 + "Metoda QR" + "-" * 31)
        self.generate_qr_plots()



M = [
    [9, 2, 0, 0],
    [2, 4, 1, 0],
    [0, 1, 3, 1],
    [0, 0, 1, 2]
]

solution = Solution(M)
solution.run_all()
eigenvalues = np.linalg.eigvals(np.array(M))

print("Rozwiązanie numpy: " + str(eigenvalues))
