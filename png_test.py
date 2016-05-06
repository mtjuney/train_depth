from PIL import Image, ImageFilter, ImageOps



def show_I(i_image):
    (width, height) = i_image.size

    o_image = Image.new(iimg.mode, i_image.size, None)

    i_pixel = i_image.load()
    o_pixel = o_image.load()


    max_value = 0
    for w in range(width):
        for h in range(height):
            if max_value < i_pixel[w, h]:
                max_value = i_pixel[w, h]


    for w in range(width):
        for h in range(height):
            o_pixel[w, h] = i_pixel[w, h] * 255 / max_value


    o_image.show()


if __name__ == '__main__':

    iimg = Image.open('data/apple_1_crop/apple_1_1_1_depthcrop.png', 'r')

    print iimg.size

    show_I(iimg)
