import cv2
import os
import numpy as np


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


def fltp(point):
    return tuple(point.astype(int).flatten())


def draw_correspondences(img, dstpoints, projpts):

    display = img.copy()
    dstpoints = norm2pix(img.shape, dstpoints, True)
    projpts = norm2pix(img.shape, projpts, True)

    for pts, color in [(projpts, (255, 0, 0)),
                       (dstpoints, (0, 0, 255))]:

        for point in pts:
            cv2.circle(display, fltp(point), 15, color, -1, cv2.LINE_AA)

    for point_a, point_b in zip(projpts, dstpoints):
        cv2.line(display, fltp(point_a), fltp(point_b),
                 (255, 255, 255), 10, cv2.LINE_AA)

    return display


def pix2norm(shape, points):
    height, width = shape[:2]
    scale = 2.0/(max(height, width))
    offset = np.array([width, height], dtype=points.dtype).reshape((-1, 1, 2))*0.5
    return (points - offset) * scale


def norm2pix(shape, points, as_integer):
    height, width = shape[:2]
    scale = max(height, width)*0.5
    offset = np.array([0.5*width, 0.5*height], dtype=points.dtype).reshape((-1, 1, 2))
    rval = points * scale + offset
    if as_integer:
        return (rval + 0.5).astype(int)
    else:
        return rval


def round_nearest_multiple(i, factor):
    i = int(i)
    rem = i % factor
    if not rem:
        return i
    else:
        return i + factor - rem
    

def print_position_list(list, x_coord):
    if len(list) == 0:
        print("(", len(list), ") ", x_coord, ": ---")
        return
  
    print("(", len(list), ") ", list[0][0], ":   ", end="")

    for i in list:
        print(i[1], ", ", end="")

    print()