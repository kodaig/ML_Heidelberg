import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
from skimage import data, img_as_float
import scipy.sparse
import time

def norm01(data):
    res = data.astype('float')
    # res -= res.min()
    res /= res.max()
    return res

def performance(p1, p2):
    return np.linalg.norm(p1 - p2)

def buildConstQ(imshape, beta = -1.):
    nrows, ncols = imshape
    size = nrows * ncols
    
    Q = scipy.sparse.lil_matrix((size,size), dtype='float')

    diagElem = 4

    def getInd(x, y):
        return x + y * ncols

    for x in range(nrows):
        for y in range(ncols):

            i = getInd(x,y)
            Q[i,i] = diagElem

            if x < ncols-1:
                j = getInd(x+1, y)
                Q[i,j] = beta
                Q[j,i] = beta

            if y < nrows-1:
                j = getInd(x, y+1)
                Q[i,j] = beta
                Q[j,i] = beta

    print("Const Q built!")

    return Q.tocsc()

def buildFancyQ(im, gamma, alpha=-1.):
    #print("alpha:",alpha,"gamma:",gamma)
    nrows, ncols, ch = im.shape
    size = nrows * ncols
    
    Q = scipy.sparse.lil_matrix((size,size), dtype='float')

    def getBeta(c0, c1):
        return alpha * np.exp(-gamma * np.linalg.norm(c0 - c1))

    def getInd(x, y):
        return x + y * ncols

    for x in range(nrows):
        for y in range(ncols):

            c0 = im[x,y]
            i = getInd(x,y)

            if x < ncols-1:
                beta = getBeta(c0, im[x+1, y])
                j = getInd(x+1, y)
                Q[i,j] = beta
                Q[j,i] = beta
                Q[i,i] += abs(beta)
                Q[j,j] += abs(beta)

            if y < nrows-1:
                beta = getBeta(c0, im[x, y+1])
                j = getInd(x, y+1)
                Q[i,j] = beta
                Q[j,i] = beta
                Q[i,i] += abs(beta)
                Q[j,j] += abs(beta)
        
    # for i in range(size):
    #     Q[i,i] = abs(np.sum(Q[i]))
    
    print("Fancy Q built!")

    return Q.tocsc()

def denoise(img, Q, sigma = 1.0):
    shape = img.shape
    size = shape[0] * shape[1]
    
    # init result image
    result = np.zeros(shape)

    # (I + sigma^2 Q) z = x
    A = scipy.sparse.identity(size, format='csc') + sigma**2 * Q

    for c in range(3):
        x = img[:,:,c].ravel()

        z = scipy.sparse.linalg.spsolve(A, x)

        result[:,:,c] = z.reshape(shape[0:2])
        
        print("Solved for color ", c)

    return result


# parameters
noise = 0.1
sigma = 1.0
gamma = 3.

start = time.time()
# load image
gt_img = ski.data.astronaut()
gt_img = norm01(gt_img)

# get image parameters
shape = gt_img.shape
size = shape[0] * shape[1]

# add noise
noisy_img = ski.util.random_noise(gt_img, var=noise**2)
noisy_img = np.clip(noisy_img, 0, 1)


# constQ = buildConstQ(shape[0:2], -.5)
fancyQ = buildFancyQ(noisy_img, gamma)


# result_img = denoise(noisy_img, constQ, sigma)
result_img = denoise(noisy_img, fancyQ, sigma)
result_img = norm01(result_img)

print("Time from loading image to denoised image: ", time.time() - start)


plt.subplot(131)
plt.imshow(gt_img)
plt.title("Original")

plt.subplot(132)
plt.imshow(noisy_img)
plt.title("Noisy")

plt.subplot(133)
plt.imshow(result_img)
plt.title("Denoised")

plt.suptitle(performance(gt_img, result_img))

plt.savefig("result.png")
