from __future__ import print_function

import csv
import os
from PIL import Image


train_path = 'data/train/'
val_path = 'data/val/'


train_list = []
with file(train_path + 'list.csv') as f:
    readcsv = csv.reader(f)
    for row in readcsv:
        train_list.append(row)

val_list = []
with file(val_path + 'list.csv') as f:
    readcsv = csv.reader(f)
    for row in readcsv:
        val_list.append(row)

for x_name, y_name in train_list:
    image = Image.open(x_name).transpose(Image.ROTATE_270)
    image.save(x_name)

for x_name, y_name in val_list:
    image = Image.open(x_name).transpose(Image.ROTATE_270)
    image.save(x_name)
