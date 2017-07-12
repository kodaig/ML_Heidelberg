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
    return sigmoid(np.sin(phi) * x[:,0] + np.cos(phi) * x[:,1] + r)

def derivative(x, phi, r):
    return sig_deriv(np.sin(phi) * x[:,0] + np.cos(phi) * x[:,1] + r)

def dzdp(x, phi, r):
    return (np.cos(phi) * x[:,0] - np.sin(phi) * x[:,1]) * derivative(x, phi, r)

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


# Losses
la = np.zeros((N,M))
lb = np.zeros((N,M))
fpp = np.zeros((N,M))
frr = np.zeros((N,M))
fpr = np.zeros((N,M))

for ip, phi in enumerate(phi_list):
    for ir,r in enumerate(r_list):
        za = out(a[:,], phi, r)
        zb = out(b[:,], phi, r)
        la[ip, ir] = loss(za, np.zeros(NData))
        lb[ip, ir] = loss(zb, np.ones(NData))

        fpp[ip, ir] = fischerPP(a[:,], phi, r) + fischerPP(b[:,], phi, r)
        frr[ip, ir] = fischerRR(a[:,], phi, r) + fischerRR(b[:,], phi, r)
        fpr[ip, ir] = fischerPR(a[:,], phi, r) + fischerPR(b[:,], phi, r)

# Total loss
l = la + lb

results = [l, fpp, frr, fpr]
titles = ["Loss", "$F_{\phi \phi}$", "$F_{rr}$", "$F_{\phi r}$"]

fig, ax = plt.subplots(2, 2)

for i, a in enumerate(ax.ravel()):
    a.imshow(results[i], origin='lower', extent=[-5, 5, 0, 2*np.pi], aspect='auto')
    a.set_title(titles[i])
    a.set_xlabel("r")
    a.set_ylabel("phi")


fig.subplots_adjust(wspace=0.3, hspace=0.3)
plt.show()
# plt.subplot(221)
# plt.imshow(l)
# plt.colorbar()
# plt.title("Loss")
# plt.xlabel("r")
# plt.ylabel("phi")

# plt.subplot(222)
# plt.imshow(fpp)
# plt.colorbar()
# plt.title("$F_{\phi \phi}$")
# plt.xlabel("r")
# plt.ylabel("phi")

# plt.subplot(223)
# plt.imshow(frr)
# plt.colorbar()
# plt.title("$F_{rr}$")
# plt.xlabel("r")
# plt.ylabel("phi")

# plt.subplot(224)
# plt.imshow(fpr)
# plt.colorbar()
# plt.title("$F_{\phi r}$")
# plt.xlabel("r")
# plt.ylabel("phi")

# plt.show()
