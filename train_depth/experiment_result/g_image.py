from __future__ import print_function
import argparse
import datetime
import json
import multiprocessing
import random
import sys
import threading
import time
import os
import yaml
import csv
import colorsys
import glob
import shutil

import numpy as np
from PIL import Image
import six
import six.moves.cPickle as pickle
from six.moves import queue

# from chainer import computational_graph as c
import chainer
from chainer import cuda
from chainer import optimizers

from readmat import readDepthMap

from neuralnet import Net



parser = argparse.ArgumentParser(description='Learning convnet from ILSVRC2012 dataset')
parser.add_argument('--logdir', '-m', default='result/000010_success/', help='NeuralNetwork model')
parser.add_argument('--genenum', '-g', default=20, type=int, help='Number of generate file')
parser.add_argument('--width', '-w', default=86, type=int, help='width')
parser.add_argument('--height', '-t', default=107, type=int, help='height')

args = parser.parse_args()


sys.path.append('neuralnet/')

NORM_MAX = 81

model = None
with open(args.logdir + 'model', 'rb') as f:
    model = pickle.load(f)

config = None
with open(args.logdir + 'config.yml', 'r') as f:
    config = yaml.load(f)


SIZE = WIDTH_IMG = HEIGHT_IMG = None
if args.width == 0 and args.height == 0:
    SIZE_IMG = (WIDTH_IMG, HEIGHT_IMG) = (config['WIDTH_IMG'], config['HEIGHT_IMG'])
else:
    SIZE_IMG = (WIDTH_IMG, HEIGHT_IMG) = (args.width, args.height)

if not glob.glob('g_images'):
    os.mkdir('g_images')

i = 0
while(glob.glob('g_images/{:0>6}*'.format(i))):
    i += 1
RESULT_DIRNAME = 'g_images/{:0>6}'.format(i)
os.mkdir(RESULT_DIRNAME)
result_path = 'g_images/{:0>6}/'.format(i)


config_write = {'logdir':args.logdir, 'WIDTH_IMG':WIDTH_IMG, 'HEIGHT_IMG':HEIGHT_IMG, 'genenum':args.genenum}
with open(result_path + 'config_g.yml', 'w') as f:
    f.write(yaml.dump(config_write))

shutil.copy(args.logdir + 'config.yml', result_path + 'config.yml')


path_val = '../data/val/'
val_list = []
with open(path_val + 'list.csv') as f:
    readcsv = csv.reader(f)
    for row in readcsv:
        val_list.append(row)



def f2t(in_image):

    out_image = Image.new('RGB', in_image.size, 'white')
    (width, height) = in_image.size

    in_pix = in_image.load()
    out_pix = out_image.load()


    for w in range(width):
        for h in range(height):
            x = in_pix[w, h] / NORM_MAX
            x = min(x, 1.0)
            x = max(x, 0.0)
            x = 1.15 - (x * 0.65)
            r, g, b = colorsys.hsv_to_rgb(x, 1.0, 1.0)
            r = int(r * 255)
            g = int(g * 255)
            b = int(b * 255)

            out_pix[w, h] = (r, g, b)

    return out_image


for i in range(args.genenum):
    x_name, y_name = val_list[i]
    x_image = Image.open('../' + x_name).resize((WIDTH_IMG, HEIGHT_IMG))
    y_image = readDepthMap('../' + y_name).resize((WIDTH_IMG, HEIGHT_IMG))

    y_image_pix = y_image.load()
    for w in range(WIDTH_IMG):
        for h in range(HEIGHT_IMG):
            y_image_pix[w, h] = y_image_pix[w, h] * 255. / NORM_MAX

    y_image_t = f2t(y_image)

    x_array_pre = np.asarray(x_image.convert("YCbCr"), dtype=np.float32).transpose(2, 0, 1) / 255
    x_array = np.ndarray((1,) + x_array_pre.shape, dtype=np.float32)
    x_array[0] = x_array_pre.copy()

    x = chainer.Variable(x_array, volatile=True)
    out_array = model.forward_super(x, train=False).data * 255
    out_array = out_array[0].reshape((HEIGHT_IMG, WIDTH_IMG))
    out_image = Image.fromarray(out_array, mode='F')

    out_image_t = f2t(out_image)


    x_image.save(result_path + '{:0>2}_color.png'.format(i))
    y_image_t.save(result_path + '{:0>2}_depth_truth.png'.format(i))
    out_image_t.save(result_path + '{:0>2}_depth_out.png'.format(i))




os.rename(RESULT_DIRNAME, RESULT_DIRNAME + '_success')
