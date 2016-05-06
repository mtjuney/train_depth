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

import numpy as np
from PIL import Image
import six
import six.moves.cPickle as pickle
from six.moves import queue

# from chainer import computational_graph as c
from chainer import cuda
from chainer import optimizers




parser = argparse.ArgumentParser(description='Learning convnet from ILSVRC2012 dataset')
parser.add_argument('--image', '-i', default='defaultimage.jpg', help='Choose image file')
parser.add_argument('--model', '-m', default='../log/000026_sucsess/model', help='NeuralNetwork model')
parser.add_argument('--width', '-w', default=30, type=int, help='Width of Image')
parser.add_argument('--height', '-t', default=30, type=int, help='Height of Image')
parser.add_argument('--noshow', '-n', default=0, type=int, help='image dont show')

args = parser.parse_args()

(WIDTH_IMG, HEIGHT_IMG) = (args.width, args.height)

image_path = "image/"
image = Image.open(image_path + args.image).resize((WIDTH_IMG, HEIGHT_IMG)).transpose(Image.ROTATE_270)


image_in = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)


sys.path.append('../neuralnet/')
model = None
with open(args.model, 'rb') as f:
    model = pickle.load(f)

image_in_array = np.ndarray((1,) + image_in.shape, dtype=np.float32)
image_in_array[0] = image_in.copy()

image_in_array = image_in_array / 255
image_out_array = model.forward_result(image_in_array).data * 255

image_out_array2 = image_out_array[0].reshape((HEIGHT_IMG, WIDTH_IMG))
image_out = Image.fromarray(image_out_array2, mode='F')
image_out_t = Image.new('RGB', (WIDTH_IMG, HEIGHT_IMG), 'white')

image_out_pix = image_out.load()
image_out_t_pix = image_out_t.load()
for w in range(WIDTH_IMG):
    for h in range(HEIGHT_IMG):
        x = image_out_pix[w, h]
        r, g, b = colorsys.hsv_to_rgb(x / 255, 1.0, 1.0)
        r = int(r * 255)
        g = int(g * 255)
        b = int(b * 255)

        image_out_t_pix[w, h] = (r, g, b)

if args.noshow <= 0:
    image.show()
    image_out_t.show()

image_out_t.save('result/' + args.image)
