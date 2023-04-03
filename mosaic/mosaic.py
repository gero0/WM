from math import ceil
from PIL import Image
import numpy as np

cmap = {
    'R': 0,
    'G': 1,
    'B': 2,
}

window1 = np.array([['G', 'B'], ['R', 'G']])
window2 = np.array([
    ['G', 'B', 'R', 'G', 'R', 'B'],
    ['R', 'G', 'G', 'B', 'G', 'G'],
    ['B', 'G', 'G', 'R', 'G', 'G'],
    ['G', 'R', 'B', 'G', 'B', 'R'],
    ['B', 'G', 'G', 'R', 'G', 'G'],
    ['R', 'G', 'G', 'B', 'G', 'G'],
])

invWindow1 = np.array([[0.5, 1], [1, 0.5]])

def make_mosaic(img, window):
    mosaic = np.zeros(img.shape, dtype=np.uint8)
    (h, w, _) = img.shape
    for y in range(h):
        for x in range(w):
            (w_h, w_w) = window.shape
            win_value = cmap[window[y % w_h, x % w_w]]
            mosaic[y, x, win_value] = img[y, x, win_value]

    return mosaic

# def demosaic_scaledown(mosaic, window):
#     (m_h, m_w, _) = mosaic.shape
#     (w_h, w_w) = window.shape
#     (h, w) = ceil(m_h / w_h), ceil(m_w / w_w)

#     img = np.zeros((h,w), dtype=np.uint8)

#     # iterate over pixels of new image...
#     for y in range(h):
#         for x in range(w):
#             #multiply pixels from original by window element-wise
#             for w_y in range(w_h):
#                 for w_x in range(w_w):

#                     (y_offset, x_offset) = (y * w_h + w_y, x + w_w * w_x)
#                     if(y_offset )
                    

#     return img

pic = Image.open("test3.png")
img = np.array(pic, dtype=np.uint8)
mosaic = make_mosaic(img, window2)


result = Image.fromarray(mosaic)
result.save("output.png")

# demosaic_scaledown(mosaic, invWindow1)
