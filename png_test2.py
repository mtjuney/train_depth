from PIL import Image, ImageFilter, ImageOps

import chainer.functions as F


if __name__ == '__main__':

    iimg = Image.open('data/rgbd-dataset/apple/apple_1/apple_1_1_1_depthcrop.png', 'r')

    ipix = iimg.load()

    (width, height) = iimg.size

    a = []

    for w in range(width):
        for h in range(height):
            if ipix[w, h] not in a:
                a.append(ipix[w, h])


    print width * height
    width_o = len(a)

    sorted_a = sorted(a)
    print sorted_a

    oimg = Image.new('I', (width_o, 1), 0)

    opix = oimg.load()

    for w in range(width_o):
        opix[w, 0] = sorted_a[w] * 256 / 1165

    oimg.show()
