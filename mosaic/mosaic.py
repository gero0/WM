from PIL import Image
import numpy as np
import itertools as it

# 'R': 0,
# 'G': 1,
# 'B': 2,
window_bayer = np.array([[1, 0], [2, 1]])
window_xtrans = np.array([
    [1, 2, 0, 1, 0, 2],
    [0, 1, 1, 2, 1, 1],
    [2, 1, 1, 0, 1, 1],
    [1, 0, 2, 1, 2, 0],
    [2, 1, 1, 0, 1, 1],
    [0, 1, 1, 2, 1, 1],
])


def make_mask(shape, window):
    h, w, _ = shape
    wh, ww = window.shape
    mask = np.zeros(shape, dtype=np.uint8)
    for y, x in it.product(range(h), range(w)):
        mask[y, x, window[y % wh, x % ww]] = 1

    return mask


def make_mosaic(img, mask):
    return img * mask


# def find_point_left(i, data, mask):
#     #cast to make sure integer is signed
#     pos = int(i) - 1

#     while (pos >= 0):
#         if mask[pos]:
#             return pos, data[pos]
#         pos -= 1
#     return -1, 0


def find_point_right(i, data, mask):
    pos = i + 1

    while (pos < len(data)):
        if mask[pos]:
            return pos, data[pos]
        pos += 1

    return len(data), 0


def linear(data, mask):
    x = 0
    (x0, y0), (x1, y1) = (-1, 0), find_point_right(x, data, mask)
    if (x1 == len(data)):
        #No data points found! This is expected in some cases. Perform no interpolation
        return data, mask

    while x < len(data):
        # Mask marks if data for this point is provided in input
        # data-point with given value, start new interpolation range
        if (mask[x] == 1):
            (x0, y0) = x, data[x]
            (x1, y1) = find_point_right(x, data, mask)
        else:
            # Interpolate
            data[x] = (y0 * (x1 - x) + y1 * (x - x0)) / (x1 - x0)
            # Mark point as containing value, so it can be used in next stage of 2D interpolation
            mask[x] = 1
        x += 1
    return data, mask


def nn(data, mask):
    x = 0
    (x0, y0), (x1, y1) = (-1, 0), find_point_right(x, data, mask)
    if (x1 == len(data)):
        #No data points found! This is expected in some cases. Perform no interpolation
        return data, mask
    
    halfpoint = (x1 - x0) / 2

    while x < len(data):
        # Mask marks if data for this point is provided in input
        # data-point with given value, start new interpolation range
        if (mask[x] == 1):
            (x0, y0) = x, data[x]
            (x1, y1) = find_point_right(x, data, mask)
            halfpoint = (x1 - x0) / 2
        else:
            # Interpolate
            if (x >= halfpoint):
                data[x] = y1
            else:
                data[x] = y0
            # Mark point as containing value, so it can be used in next stage of 2D interpolation
            mask[x] = 1
        x += 1
    return data, mask

def bilinear_interpolation(mosaic, mask, interp_f):
    output = np.zeros(mosaic.shape, dtype=np.uint8)
    h, w, _ = mosaic.shape

    #Perform interpolation on each color channel
    for channel in range(3):
        #Interpolate by rows
        for row in range(h):
            interp_row, interp_mask = interp_f(mosaic[row, :, channel],
                                               mask[row, :, channel])
            output[row, :, channel] = interp_row
            mask[row, :, channel] = interp_mask

        #Then by columns. Note we're using previous stage's output as input here
        for column in range(w):
            interp_col, _mask = interp_f(output[:, column, channel],
                                         mask[:, column, channel])
            output[:, column, channel] = interp_col

    return output


pic = Image.open("test2.jpg")
img = np.array(pic, dtype=np.uint8)

mask = make_mask(img.shape, window_bayer)
mosaic = make_mosaic(img, mask)

result = Image.fromarray(mask * 255)
result.save("mask.png")

result = Image.fromarray(mosaic)
result.save("output.png")

output = bilinear_interpolation(mosaic, mask, nn)
output = Image.fromarray(output)
output.save("interpolated.png")
