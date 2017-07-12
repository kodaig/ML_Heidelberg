import numpy as np
import matplotlib.pyplot as plt


# Entropy loss
def loss(z, gt):
    n = z.size
    l = gt * np.log(z) + (1 - gt) * np.log(1 - z)
    return - sum(l) / n

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sig_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Our neural node
def out(x, phi, r):
    return sigmoid(np.sin(phi) * x[0] + np.cos(phi) * x[1] + r)

def derivative(x, phi, r):
    return sig_deriv(np.sin(phi) * x[0] + np.cos(phi) * x[1] + r)

def dzdp(x, phi, r):
    return (np.cos(phi) * x[0] - np.sin(phi) * x[1]) * derivative(x, phi, r)

def dzdr(x, phi, r):
    return derivative(x, phi, r)

def fischerPP(x, phi, r):
    z = out(x, phi, r)
    f = dzdp(x, phi, r)**2 / (z * (1 - z))
    return sum(f)

def fischerRR(x, phi, r):
    z = out(x, phi, r)
    f = dzdr(x, phi, r)**2 / (z * (1 - z))
    return sum(f)

def fischerPR(x, phi, r):
    z = out(x, phi, r)
    f = dzdp(x, phi, r) * dzdr(x, phi, r) / (z * (1 - z))
    return sum(f)



N = 50
M = 50
NData = 100
MeanA = [6, 1]
MeanB = [6, 2]
Cov = [[1,0],
       [0,1]]

phi_list = np.linspace(0, 2*np.pi, N)
r_list = np.linspace(-5.0, 5.0, M)

# Make sample data
a = np.random.multivariate_normal(MeanA, Cov, NData)
b = np.random.multivariate_normal(MeanB, Cov, NData)


# Output
za = np.zeros(NData)
zb = np.zeros(NData)

# Losses
la = lb = np.zeros((N,M))
fpp = frr = fpr = np.zeros((N,M))

for ip, phi in enumerate(phi_list):
    for ir,r in enumerate(r_list):
        for i in range(NData):
            za[i] = out(a[i,], phi, r)
            zb[i] = out(b[i,], phi, r)
        la[ip, ir] = loss(za, np.zeros(NData))
        lb[ip, ir] = loss(zb, np.ones(NData))
        fpp[ip, ir] = fischerPP(a[:,], phi, r)

# Total loss
l = la + lb



# plt.imshow(l)
plt.imshow(fpp)
plt.colorbar()

plt.xlabel("r")
plt.ylabel("phi")
plt.show()
