from __future__ import print_function

import glob
import re
import os.path
import csv

path = 'data/train/'

images = [r.split('/')[-1] for r in glob.glob(path + 'x/' + 'img-*.jpg')]

pairs = []
fails = []
for image in images:
    depth = image.replace('img-', 'depth-').replace('jpg', 'mat')
    if os.path.isfile(path + 'y/' + depth):
        pairs.append((path + 'x/' + image, path + 'y/' + depth))
    else:
        fails.append((image, depth))

with file(path + 'list_origin.csv', 'w') as f:
    writecsv = csv.writer(f, lineterminator='\n')

    writecsv.writerows(pairs)

print(fails)
