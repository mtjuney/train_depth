#!/usr/bin/env python
import argparse
import sys
import csv

import numpy
from PIL import Image
import six.moves.cPickle as pickle


parser = argparse.ArgumentParser(description='Compute images mean array')
parser.add_argument('dataset', help='Path to training image-label list file')
parser.add_argument('--output', '-o', default='mean.npy',
                    help='path to output mean array')
parser.add_argument('--width', '-w', default=86, type=int)
parser.add_argument('--height', '-t', default=107, type=int)
args = parser.parse_args()


data_list = []
with open(args.dataset) as f:
    readcsv = csv.reader(f)
    for row in readcsv:
        data_list.append(row[0])


sum_image = None
count = 0
for filepath in data_list:
    image = numpy.asarray(Image.open(filepath).resize((args.width, args.height))).transpose(2, 0, 1)
    if sum_image is None:
        sum_image = numpy.ndarray(image.shape, dtype=numpy.float32)
        sum_image[:] = image
    else:
        sum_image += image
    count += 1
    sys.stderr.write('\r{}'.format(count))
    sys.stderr.flush()

sys.stderr.write('\n')

mean = sum_image / count
pickle.dump(mean, open(args.output, 'wb'), -1)
