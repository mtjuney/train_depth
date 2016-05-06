from __future__ import print_function

from PIL import Image
import numpy as np
import csv
import colorsys

from readmat import readDepthMap

CONV_FLAG = False

def read_image(x_name, y_name):
    image_x = np.asarray(Image.open(x_name).resize((WIDTH_IMG, HEIGHT_IMG)).transpose(Image.ROTATE_270), dtype=np.float32).transpose(2, 1, 0)
    if CONV_FLAG:
        image_y = np.asarray(readDepthMap(y_name), dtype=np.float32).reshape((WIDTH_IMG, HEIGHT_IMG, 1)).transpose(2, 1, 0)
    else:
        image_y = np.asarray(readDepthMap(y_name), dtype=np.float32).reshape(-1)


    output_x = image_x / 255
    output_y = image_y / NORM_MAX
    return (output_x, output_y)



path_train = 'data/train/'
path_val = 'data/val/'

WIDTH_IMG = 500
HEIGHT_IMG = 500

train_list = []
with file(path_train + 'list_origin.csv') as f:
    readcsv = csv.reader(f)
    for row in readcsv:
        train_list.append(row)

val_list = []
with file(path_val + 'list.csv') as f:
    readcsv = csv.reader(f)
    for row in readcsv:
        val_list.append(row)

print(len(train_list))
x_name, y_name = train_list[5]

# image1 = Image.open(x_name).resize((WIDTH_IMG, HEIGHT_IMG)).transpose(Image.ROTATE_270)
# image2 = readDepthMap(y_name).resize((WIDTH_IMG, HEIGHT_IMG))

image1 = Image.open(x_name).transpose(Image.ROTATE_270)
image2 = readDepthMap(y_name)

image2_pix = image2.load()

image3 = Image.new('RGB', image2.size, 'white')
image3_pix = image3.load()
for w in range(image2.size[0]):
    for h in range(image2.size[1]):
        x = image2_pix[w, h]
        r, g, b = colorsys.hsv_to_rgb(x / (81 * 1.5), 1.0, 1.0)
        r = int(r * 255)
        g = int(g * 255)
        b = int(b * 255)

        image3_pix[w, h] = (r, g, b)


# image1 = Image.open(x_name).resize((WIDTH_IMG, HEIGHT_IMG))
# image2 = Image.new('RGB', (WIDTH_IMG, HEIGHT_IMG), 'white')
# image2_pix = image2.load()
#
# image1_array = np.asarray(image1)
# print(image2.size)
# image2_array = np.ndarray((WIDTH_IMG, HEIGHT_IMG, 3))
# for w in range(WIDTH_IMG):
#     for h in range(HEIGHT_IMG):
#         for c in range(3):
#             image2_pix[w, h] = tuple(image1_array[w, h].tolist())
# image1.save('../../../Desktop/rgb.png')
# image3.save('../../../Desktop/depth.png')

print(image1.size)
print(image3.size)

image1.show()
image3.show()
