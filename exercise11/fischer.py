import numpy as np
import matplotlib.pyplot as plt


def loss(z, gt):
    n = z.size
    l = gt * np.log(z) + (1 - gt) * np.log(1 - z)
    return - sum(l) / n

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def output(x, phi, r):
    return sigmoid(np.sin(phi) * x[0] + np.cos(phi) * x[1] + r)


N = 10
M = 10

phi_list = np.linspace(0, 2*np.pi, N)
r_list = np.linspace(-5.0, 5.0, M)


NData = 100
MeanA = [1, 1]
MeanB = [1, 2]
Cov = [[1,0],
       [0,1]]

a = np.random.multivariate_normal(MeanA, Cov, NData)
b = np.random.multivariate_normal(MeanB, Cov, NData)


za = np.zeros(NData)
zb = np.zeros(NData)

la = np.zeros((N,M))
lb = np.zeros((N,M))

for ip, phi in enumerate(phi_list):
    for ir,r in enumerate(r_list):
        for i in range(NData):
            za[i] = output(a[i,], phi, r)
            zb[i] = output(b[i,], phi, r)
        la[ip, ir] = loss(za, np.zeros(NData))
        lb[ip, ir] = loss(zb, np.ones(NData))

l = la + lb


plt.imshow(la+lb)
plt.colorbar()


plt.xlabel("phi")
plt.ylabel("r")
plt.show()
