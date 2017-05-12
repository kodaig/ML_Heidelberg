import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog


#coefficient vector
def cost(b):
    c = np.array([0.1,0.1,0.1,0.9,0.9,0.1,0.,b,b,0.,0.,b,b,0.,0.,b,b,0.])
    return c

beta = -1.0
c = cost(beta)
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
A = np. zeros((15,18))
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

b = np.zeros(15)
for i in range(3):
    b[i] = 1

bounds = (0,None)

res = linprog(c, A_eq=A, b_eq=b, bounds=(bounds), options={"disp": True})

print("beta=",beta)
print("coefficients vector c=\n",c)
print("constraint matrix A=\n",A)
print("constraint vector b=\n",b)
print("solution vector mu=\n",res.x)

# for checking
def energy(cost, state):
    return sum(cost * state)

def givemu(x0, x1, x2):
    mu = np.zeros(18)
    mu[x0] = 1
    mu[x1 + 2] = 1
    mu[x2 + 4] = 1

    mu[6] = mu[0]*mu[2]
    mu[7] = mu[0]*mu[3]
    mu[8] = mu[1]*mu[2]
    mu[9] = mu[1]*mu[3]

    mu[10] = mu[0]*mu[2]
    mu[11] = mu[0]*mu[3]
    mu[12] = mu[1]*mu[2]
    mu[13] = mu[1]*mu[3]

    mu[14] = mu[2]*mu[4]
    mu[15] = mu[2]*mu[5]
    mu[16] = mu[3]*mu[4]
    mu[17] = mu[3]*mu[5]
    return mu

def givex(mu):
    x = np.zeros(3)
    x[0] = int(mu[1] == 1)
    x[1] = int(mu[3] == 1)
    x[2] = int(mu[5] == 1)
    return x

# print(energy(c, res.x))
# print(energy(c, givemu(0,0,0)))
# print(givemu(1,1,1))
# print(energy(c, givemu(1,1,1)))
