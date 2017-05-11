import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

beta = 1.0

#coefficient vector
c = np.array([0.1,0.1,0.1,0.9,0.9,0.1,0,beta,beta,0,0,beta,beta,0,0,beta,beta,0])
"""
I set solution vector "mu" as
mu=
[mu0(0),mu0(1),mu1(0),mu1(1),mu2(0),mu2(1)
,mu01(0,0),mu01(0,1),mu01(1,0),mu01(1,1)
,mu12(0,0),mu12(0,1),mu12(1,0),mu12(1,1)
,mu20(0,0),mu20(0,1),mu20(1,0),mu20(1,1)]
mui(k) = 1 (xi=k), 0 (otherwise)
muij(k,l) = 1 ((xi,xj)=(k,l)), 0 (otherwise)
"""

#constraint matrix
A = np.array([[0]*18]*15)
for i in range(3):
    A[i,2*i] = 1
    A[i,2*i+1] = 1

for i in range(3,15):
    j = i - 3
    k = j - 2*(j//4)
    A[i,k-6*(k//6)] = -1    

a = np.array([[1,1,0,0],[0,0,1,1],[1,0,1,0],[0,1,0,1]])
A[3:7,6:10] = a
A[7:11,10:14] = a
A[11:15,14:18] = a

b = np.array([0]*15)
for i in range(3):
    b[i] = 1

bounds = (0, None)

res = linprog(c, A_eq=A, b_eq=b, bounds=(bounds), options={"disp": True})

print("beta=",beta)
print("coefficients vector c=\n",c)
print("constraint matrix A=\n",A)
print("solution vector mu=\n",res.x)
