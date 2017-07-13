import numpy as np
import matplotlib.pyplot as plt

from fischer import *

# ==============================================
# === BONUS: Gradients and Natural Gradients ===
# ==============================================

# Derivative dloss/dphi
def dldp(x, y, phi, r):
    z = nn(x, phi, r)
    dl = dzdp(x, phi, r) * (y - z) / (z * (1 - z))
    return - sum(dl) / dl.size

# Derivative dloss/dr
def dldr(x, y, phi, r):
    z = nn(x, phi, r)
    dl = dzdr(x, phi, r) * (y - z) / (z * (1 - z))
    return - sum(dl) / dl.size

def vectorfields(Ndata, mean, N, M):
    Cov = [[1,0],
           [0,1]]

    # Make sample data
    class1 = np.random.multivariate_normal(mean[0], Cov, NData)
    class2 = np.random.multivariate_normal(mean[1], Cov, NData)

    phi_list = np.linspace(0, 2*np.pi, N)
    r_list = np.linspace(-5.0, 5.0, M)

    # Gradient
    dp = np.zeros((N,M))
    dr = np.zeros((N,M))

    # Fisher Matrix
    fpp = np.zeros((N,M))
    frr = np.zeros((N,M))
    fpr = np.zeros((N,M))

    for ip, phi in enumerate(phi_list):
        for ir, r in enumerate(r_list):
            z1 = nn(class1[:,], phi, r)
            z2 = nn(class2[:,], phi, r)
            dp[ip, ir] = dldp(class1[:,], 0.0, phi, r)
            dr[ip, ir] = dldr(class2[:,], 1.0, phi, r)

            fpp[ip, ir] = fischerPP(class1[:,], phi, r) + fischerPP(class2[:,], phi, r)
            frr[ip, ir] = fischerRR(class1[:,], phi, r) + fischerRR(class2[:,], phi, r)
            fpr[ip, ir] = fischerPR(class1[:,], phi, r) + fischerPR(class2[:,], phi, r)

    fig = plt.figure()

    skip = (slice(None, None, 3), slice(None, None, 3))
    xx, yy = np.meshgrid(r_list, phi_list)

    plt.quiver(xx[skip], yy[skip], dr[skip], dp[skip], units='width')

    plt.show()


    return fig

if __name__=="__main__":
    N = 50
    M = 50
    NData = 100
    meanA = [[1, 1], [1, 2]]
    meanB = [[6, 1], [6, 2]]

    figA = vectorfields(NData, meanA, N, M)
