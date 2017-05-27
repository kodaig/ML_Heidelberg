import numpy as np
import networkx as nx
import pylab
import matplotlib.pyplot as plt

def HammingLoss(H,V):
    w, h = H.shape
    loss = 0
    for i in range(w):
        for j in range(h):
            loss += XOR((H[i,j],V[i,j]))
    return loss


def XOR(ls):
    return sum(ls)%2


def chain_solver(unary, beta):
    l = len(unary)
    e = [('s',(0,0),unary[0,0]), ('s',(0,1),unary[0,1]), ((l-1,0),'t',0), ((l-1,1),'t',0)]
    G = nx.DiGraph()
    labels = [(0,0),(0,1),(1,0),(1,1)]
    for i in range(l-1):
        for label in labels:
            e.append(((i,label[0]), (i+1,label[1]), unary[i+1,label[1]] + beta*XOR(label)))
    G.add_weighted_edges_from(e)
    shortest_path = nx.dijkstra_path(G,'s','t')
    
    ls =[]
    for i in range(1,l+1):
        ls.append(shortest_path[i][1])
    
    return np.array(ls)

# A function which optimizes E_h(X)
def E_horizontal(unary, beta):
    height, width, values = unary.shape
    E_h = []
    for i in range(height):
        E_h.append(chain_solver(unary[i,:],beta))
                   
    return np.array(E_h)

# A function which optimizes E_v(X)
def E_vertical(unary, beta):
    height, width, values = unary.shape
    E_v = []
    for i in range(width):
        E_v.append(chain_solver(unary[:,i],beta))
        
    return np.array(E_v).transpose()


betas = [0.01, 0.1, 0.2, 0.5, 1.0]
n = 20    #length of graphical model


### 1.Potts MRF with chain structure

# Give random unary energies with chain structure
unary = np.random.uniform(0,1,(n,2))
Y = np.zeros((n,2)).transpose()
Y[0,:] = unary[:,0]
Y[1,:] = unary[:,1]
plt.imshow(Y, cmap="gray", interpolation="nearest")
plt.xticks([])
plt.yticks([])
plt.show()


# Solution for different beta
for beta in betas:
    print(beta)
    X = chain_solver(unary, beta)
    Y = np.array([[0]]*n).transpose()
    Y[0,:] = X
    plt.imshow(Y, cmap="gray", interpolation="nearest")
    plt.xticks([])
    plt.yticks([])
    plt.show()


### 2. Potts MRF with grid structure

# Give random unary enegies with grid structure
unary2 = np.random.uniform(0,1,(n,n,2))
unary_h = unary2/2
unary_v = unary2/2


count = 0
# Solution for different beta
for beta in betas:
    H = E_horizontal(unary_h,beta)
    V = E_vertical(unary_v,beta)
    
    print("beta=",beta, "loss=",HammingLoss(H,V))

    f= pylab.figure()
    ax1 = f.add_subplot(1,2,1)
    pylab.imshow(H, cmap='gray', interpolation="nearest")
    ax1.set_title('Horizontal')

    ax2 = f.add_subplot(1,2,2)
    pylab.imshow(V, cmap='gray', interpolation="nearest")
    ax2.set_title('Vertical')
    plt.show()
