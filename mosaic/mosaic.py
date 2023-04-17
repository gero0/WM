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


pic = Image.open("test2.jpg")
img = np.array(pic, dtype=np.uint8)

mask = make_mask(img.shape, window_xtrans)
mosaic = make_mosaic(img, mask)

result = Image.fromarray(mask * 255)
result.save("mask.png")

result = Image.fromarray(mosaic)
result.save("output.png")
