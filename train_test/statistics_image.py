from __future__ import print_function
from PIL import Image
import numpy as np
import os
import math
import matplotlib.pylab as pl

dirpath = 'data_train/train/y/'

filenames = os.listdir(dirpath)

image = None
sum_image = 0.
max_value = None
min_value = None

means = np.ndarray(len(filenames))

for filename in filenames:
    image = Image.open(dirpath + filename, 'r')
    im = np.asarray(image).reshape(-1)

    im_max = im.max()
    im_min = im.min()

    if max_value == None:
        max_value = im_max
    elif max_value < im_max:
        max_value = im_max

    if min_value == None:
        min_value = im_min
    elif min_value > im_min:
        min_value = im_min

    sum_image += np.mean(im)

mean = sum_image / len(filenames)

sum_square = 0.
pixel_num = 0

hist = np.zeros(max_value + 1, dtype=np.int32)


for filename in filenames:
    image = Image.open(dirpath + filename, 'r')
    im = np.asarray(image).reshape(-1)

    for i in range(0, im.size):
        pixel_num += 1

        sum_square += math.pow(im[i] - mean, 2.)

        hist[im[i]] += 1

sigma = math.sqrt(sum_square / pixel_num)

print('mean', mean)
print('sigma', sigma)
print('max', max_value)
print('min', min_value)

pl.bar(np.arange(hist.size), hist)
pl.show()
