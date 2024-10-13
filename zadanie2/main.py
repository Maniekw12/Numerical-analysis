import numpy as np




#A1 = np.array([[2.40827208, -0.36066254, 0.80575445, 0.46309511, 1.20708553])
#
#
#A2 = np.array([[2.61370745, -0.6334453, 0.76061329, 0.24938964, 0.82783473],
#              [-0.6334453, 1.51060349, 0.08570081, 0.31048984, -0.53591589],
#              [0.76061329, 0.08570081, 2.46956812, 0.18519926, 0.13060923],
#              [0.24938964, 0.31048984, 0.18519926, 2.27845311, -0.54893124],
#              [0.82783473, -0.53591589, 0.13060923, -0.54893124, 2.6276678]])
#
#
#b = np.array([5.40780228, 3.67008677, 3.12306266, -1.11187948, 0.54437218])
#
#
#bp = np.array([0.000001, 0, 0, 0, 0])
#bp = bp + b
#
#
#y1 = np.linalg.solve(A1, b)
#y2 = np.linalg.solve(A2, b)
#yp1 = np.linalg.solve(A2, bp)
#yp2 = np.linalg.solve(A2, bp)
#
#
#
#print("Equations without perturbation")
#print("y1: ", y1)
#print("y2: ", y2)
#print("")
#
#print("Equations with perturbation")
#print("y1: ", yp1)
#print("y2: ", yp2)
#print("")
#
#
#d1 = np.linalg.norm(y1-yp1)
#d2 = np.linalg.norm(y2-yp2)
#
#print("delta1: ", d1)
#
#d2='{:.20f}'.format(d2)
#
#print("delta2: ", d2)
#print('\n')
#
#
#print("matrix conditioning coefficient for A1: ")
#print(np.linalg.cond(A1))
#print("matrix conditioning coefficient for A2: ")
#print(np.linalg.cond(A2))
#
#
#