import numpy as np
import matplotlib.pyplot as plt


# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid
def sig_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Our neural network
def nn(x, phi, r):
    return sigmoid(np.sin(phi) * x[:,0] + np.cos(phi) * x[:,1] + r)

# Derivative of nn
def derivative(x, phi, r):
    return sig_deriv(np.sin(phi) * x[:,0] + np.cos(phi) * x[:,1] + r)

def dzdp(x, phi, r):
    return (np.cos(phi) * x[:,0] - np.sin(phi) * x[:,1]) * derivative(x, phi, r)

def dzdr(x, phi, r):
    return derivative(x, phi, r)

def fischerPP(x, phi, r):
    z = nn(x, phi, r)
    f = dzdp(x, phi, r)**2 / (z * (1 - z))
    return sum(f)

def fischerRR(x, phi, r):
    z = nn(x, phi, r)
    f = dzdr(x, phi, r)**2 / (z * (1 - z))
    return sum(f)

def fischerPR(x, phi, r):
    z = nn(x, phi, r)
    f = dzdp(x, phi, r) * dzdr(x, phi, r) / (z * (1 - z))
    return sum(f)


# Entropy loss function
def loss(z, gt):
    n = z.size
    l = gt * np.log(z) + (1 - gt) * np.log(1 - z)
    return - sum(l) / n


def makefig(NData, mean, N, M, dataset=""):
    Cov = [[1,0],
           [0,1]]
    # Make sample data
    class1 = np.random.multivariate_normal(mean[0], Cov, NData)
    class2 = np.random.multivariate_normal(mean[1], Cov, NData)


    phi_list = np.linspace(0, 2*np.pi, N)
    r_list = np.linspace(-5.0, 5.0, M)


    # Losses
    l1 = np.zeros((N,M))
    l2 = np.zeros((N,M))
    # Fischer matrix entries
    fpp = np.zeros((N,M))
    frr = np.zeros((N,M))
    fpr = np.zeros((N,M))

    for ip, phi in enumerate(phi_list):
        for ir, r in enumerate(r_list):
            z1 = nn(class1[:,], phi, r)
            z2 = nn(class2[:,], phi, r)
            l1[ip, ir] = loss(z1, np.zeros(NData))
            l2[ip, ir] = loss(z2, np.ones(NData))

            fpp[ip, ir] = fischerPP(class1[:,], phi, r) + fischerPP(class2[:,], phi, r)
            frr[ip, ir] = fischerRR(class1[:,], phi, r) + fischerRR(class2[:,], phi, r)
            fpr[ip, ir] = fischerPR(class1[:,], phi, r) + fischerPR(class2[:,], phi, r)
    # Total loss
    l = l1 + l2


    # Make heatmaps
    results = [l, fpp, frr, fpr]
    titles = ["Loss", "$F_{\phi \phi}$", "$F_{rr}$", "$F_{\phi r}$"]

    fig, ax = plt.subplots(2, 2)
    for i, a in enumerate(ax.ravel()):
        a.imshow(results[i], origin='lower', extent=[-5, 5, 0, 2*np.pi], aspect=5/np.pi)
        a.set_title(titles[i])
        a.set_xlabel("r")
        a.set_ylabel("$\phi$")
    fig.suptitle(f"Heatmaps for dataset {dataset}")
    fig.tight_layout()

    return fig

N = 50
M = 50
NData = 100
meanA = [[1, 1], [1, 2]]
meanB = [[6, 1], [6, 2]]

figA = makefig(NData, meanA, N, M, "A")
figA.savefig("heatmaps_a.png", bbox_inches='tight')
figB = makefig(NData, meanB, N, M, "B")
figB.savefig("heatmaps_b.png", bbox_inches='tight')

# plt.show()
