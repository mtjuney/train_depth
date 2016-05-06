from __future__ import print_function
import argparse
import datetime
import json
import multiprocessing
import random
import sys
import threading
import time
import colorsys

import numpy as np
from PIL import Image
import six
import six.moves.cPickle as pickle
from six.moves import queue

from chainer import computational_graph as c
from chainer import cuda
from chainer import optimizers


parser = argparse.ArgumentParser(description='Learning convnet from ILSVRC2012 dataset')
parser.add_argument('--model', '-m', help='Path to model')
args = parser.parse_args()

sys.path.append('neuralnet')

with open(args.model, 'rb') as f:
    model = pickle.load(f)

def f2t(in_image):

    out_image = Image.new('RGB', in_image.size, 'white')
    (width, height) = in_image.size

    in_pix = in_image.load()
    out_pix = out_image.load()

    max_x = in_pix[0, 0]
    min_x = in_pix[0, 0]

    for w in range(width):
        for h in range(height):
            x = in_pix[w, h]
            if max_x < x:
                max_x = x
            if min_x > x:
                min_x = x

    for w in range(width):
        for h in range(height):
            x = (in_pix[w, h] - min_x) / ((max_x - min_x) * 2)
            # x = in_pix[w, h] / 510
            r, g, b = colorsys.hsv_to_rgb(x, 1.0, 1.0)
            r = int(r * 255)
            g = int(g * 255)
            b = int(b * 255)

            out_pix[w, h] = (r, g, b)

    return out_image





parameters = model.parameters

fltr = parameters[0][0, 0]
fltr = (fltr - fltr.min()) * 255 / (fltr.max() - fltr.min())

fltr_img = Image.fromarray(fltr, mode='F')


fltr_img.resize((900, 900)).show()
