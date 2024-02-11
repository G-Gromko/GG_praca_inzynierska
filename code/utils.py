import cv2
import os


def print_list(list):
    for i in list:
        print(i)

def debug_show(name, step, text, display):
    path = '.\img_dump'
    filetext = text.replace(' ', '_')
    outfile = name + '_debug_' + str(step) + '_' + filetext + '.png'
    cv2.imwrite(os.path.join(path, outfile), display)


def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dimension = None
    (y, x) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        ratio = height / float(y)
        dimension = (int(x * ratio), height)
    else:
        ratio = width / float(x)
        dimension = (width, int(y * ratio))

    return cv2.resize(image, dimension, interpolation=inter)