import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
from skimage import data, img_as_float
import scipy.sparse

def norm01(data):
    res = data.astype('float')
    res -= res.min()
    res /= res.max()
    return res

def buildConstQ(imshape, beta = -1.):
    diagElem = 4

    nrows, ncols = imshape
    size = nrows * ncols
    
    Q = scipy.sparse.lil_matrix((size,size), dtype='float')

    def getInd(x, y):
        return x + y * ncols

    # def getAdj(x, y):
    #     adj = []
    #     if x < ncols-1:
    #         adj.append(getInd(x+1, y))
    #     if y < nrows-1:
    #         adj.append(getInd(x, y+1))
    #     return adj


    row = []
    col = []
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

    print("Q built!")

    return Q.tocsc()

def buildQ(im, alpha, gamma):
    return 0


# load image
gt_img = ski.data.astronaut()
gt_img = norm01(gt_img)

# get image parameters
shape = gt_img.shape
size = shape[0]**2

# add noise
noise = 0.1
noisy_img = ski.util.random_noise(gt_img, var=noise**2)

noisy_img = np.clip(noisy_img, 0, 1)

# init result image
result_img = np.zeros(shape)

sigma = 0.1



# built matrizes
identity = scipy.sparse.identity(size)
constQ = buildConstQ(shape[0:2], -.5)
mat = identity + sigma**2 * constQ

for c in range(3):
    x = noisy_img[:,:,c].flatten()

    z = scipy.sparse.linalg.spsolve(mat, x)

    result_img[:,:,c] = z.reshape(shape[0:2])
    
    print("Solved for color ", c)

result_img = norm01(result_img)


plt.subplot(131)
plt.imshow(gt_img)
plt.title("Original")

plt.subplot(132)
plt.imshow(noisy_img)
plt.title("Noisy")

plt.subplot(133)
plt.imshow(result_img)
plt.title("Denoised")

plt.show()
