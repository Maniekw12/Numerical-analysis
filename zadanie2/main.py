import numpy as np

A1 = np.array([
    [5.8267103432, 1.0419816676, 0.4517861296, -0.2246976350, 0.7150286064],
    [1.0419816676, 5.8150823499, -0.8642832971, 0.6610711416, -0.3874139415],
    [0.4517861296, -0.8642832971, 1.5136472691, -0.8512078774, 0.6771688230],
    [-0.2246976350, 0.6610711416, -0.8512078774, 5.3014166511, 0.5228116055],
    [0.7150286064, -0.3874139415, 0.6771688230, 0.5228116055, 3.5431433879]])

A2 = np.array([
    [5.4763986379, 1.6846933459, 0.3136661779, -1.0597154562, 0.0083249547],
    [1.6846933459, 4.6359087874, -0.6108766748, 2.1930659258, 0.9091647433],
    [0.3136661779, -0.6108766748, 1.4591897081, -1.1804364456, 0.3985316185],
    [-1.0597154562, 2.1930659258, -1.1804364456, 3.3110327980, -1.1617171573],
    [0.0083249547, 0.9091647433, 0.3985316185, -1.1617171573, 2.1174700695]])



b = np.array([-2.8634904630, -4.8216733374, -4.2958468309, -0.0877703331, -2.0223464006]).reshape(-1, 1)

"""Rozwiazauje uklady dla niezaburzonych form macierzy"""""
y1 = np.linalg.solve(A1, b)
y2 = np.linalg.solve(A2, b)

desired_norm = 1e-6
delta_b = np.random.randn(5)
scaled_delta_b = delta_b * (desired_norm / np.linalg.norm(delta_b))
b_perturbed = b + scaled_delta_b.reshape(-1, 1)


y1_perturbed = np.linalg.solve(A1, b_perturbed)  # rozwiazuje zaburzone rownania
y2_perturbed = np.linalg.solve(A2, b_perturbed)  #
print("uwarunkowanie macierzy 1: "+ str(np.linalg.cond(A1)))
print("uwarunkowanie macierzy 2: "+ str(np.linalg.cond(A2)))

print("Wynik dla A1y=b\n")
print(y1)
print("Wynik dla A2y=b\n")
print(y2)
print("------------Wyniki dla zaburzonych wartosci------------")
print("Wynik dla A1y=b\n")
print(y1_perturbed)
print("Wynik dla A2y=b\n")
print(y2_perturbed)
