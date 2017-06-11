from __future__ import print_function, division, absolute_import

import numpy
import random 

import sys



import skimage.transform
import matplotlib.pyplot as plt



def rot_gt(gt, a):
    fgt = gt.astype(numpy.float32)
    mi,ma = fgt.min(), fgt.max()
    fgt -= mi 
    fgt /= (ma - mi)

    rot = skimage.transform.rotate(fgt,int(a), order=0)
    rot = numpy.clip(rot, 0.0, 1.0)
    rot *= (ma - mi)
    rot += mi
    return rot

def make_toy_dataset(shape=None, n_images=20, noise=1.0):
    imgs = []
    gts = []
    if shape is None:
        shape = (20, 20)
    for i in range(n_images):

        gt_img = numpy.zeros(shape)
        gt_img[0:shape[0]//2,:] = 1

        #gt_img[shape[0]//4: 3*shape[0]//4, shape[0]//4: 3*shape[0]//4]  = 2

        ra = numpy.random.randint(180)
        #print ra 
        gt_img = rot_gt(gt_img, ra)


        # plt.imshow(gt_img)
        # plt.show()

        img = gt_img + (numpy.random.random(shape)-0.5)*float(noise)

        # plt.imshow(img.squeeze())
        # plt.show()

        imgs.append(img.astype('float32'))
        gts.append(gt_img.astype('uint8'))

    return imgs, gts
