import numpy as np
import scipy
import skimage
import skimage.data
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, lil_matrix
import time

def makeQ_simple(im):
    m,n,e = im.shape
    L=m*n
    Q=lil_matrix((L,L))

    for i in range(L):
        Q[i,i]=4

    for i in range(L-1):
        Q[i,i+1]=-1
        Q[i+1,i]=-1

    for i in range(L-n):
        Q[i,i+n]=-1
        Q[i+n,i]=-1

    return Q.tocsr()

def makeQ_fancy(im,gamma,alpha=-1):
    start = time.time()
    #print("alpha:",alpha,"gamma:",gamma)
    m,n,ch = im.shape
    L = m*n
    im = im.reshape((L,3))
    Q = lil_matrix((L,L))

    for i in range(L-1):
        diff = np.linalg.norm(im[i+1]-im[i])
        C = alpha*np.exp(-gamma*diff)
        Q[i,i+1] = C
        Q[i+1,i] = C

    for i in range(L-n):
        diff = np.linalg.norm(im[i+n]-im[i])
        C = alpha*np.exp(-gamma*diff)
        Q[i,i+n] = C
        Q[i+n,i] = C

    for i in range(L):
        Q[i,i] = abs(np.sum(Q[i]))

    print("Fancy Q built in", time.time()-start)

    return Q.tocsr()

def normalize(im):
    return im/im.max()

#pic = normalize(skimage.data.astronaut()[70:150,190:270])
pic = normalize(skimage.data.astronaut())
pic_noisy = skimage.util.random_noise(pic, mode="gaussian", var=0.01)
height, width, e = pic.shape
L = height*width

#Denoising with simple Q
sigma = 0.7

A = sigma**2 * makeQ_simple(pic) + scipy.sparse.identity(L,format="csr")

pic_denoised = []
for i in range(3):
    b = pic_noisy[:,:,i].reshape((L,))
    x  =  scipy.sparse.linalg.spsolve(A,b)
    pic_denoised.append(x.reshape((L,1)))

pic_denoised = np.concatenate(pic_denoised,axis=1)
pic_denoised = pic_denoised.reshape((height,width,3))

print("noisy","Sum Squared Defference=",np.linalg.norm(pic-pic_noisy)/L)
plt.imshow(pic_noisy, interpolation="nearest")
plt.show()

print("denoised","Sum Squared Defference=",np.linalg.norm(pic-pic_denoised)/L)
plt.imshow(pic_denoised, interpolation="nearest")
plt.show()

#Denoising with fancy Q
sigma = 1.1
alpha = -1
gamma = 3

A = sigma**2*makeQ_fancy(pic,gamma) + scipy.sparse.identity(L,format="csr")

pic_denoised = []
for i in range(3):
    b = pic_noisy[:,:,i].reshape((L,))
    x = scipy.sparse.linalg.spsolve(A,b)
    pic_denoised.append(x.reshape((L,1)))

pic_denoised=np.concatenate(pic_denoised,axis=1)
pic_denoised=pic_denoised.reshape((height,width,3))

print("noisy","Sum Squared Defference=",np.linalg.norm(pic-pic_noisy)/L)
plt.imshow(pic_noisy, interpolation="nearest")
plt.show()

print("denoised","Sum Squared Defference=",np.linalg.norm(pic-pic_denoised)/L)
plt.imshow(pic_denoised, interpolation="nearest")
plt.show()
