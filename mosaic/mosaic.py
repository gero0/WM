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


def find_point_right(i, data, mask, default_value=0):
    pos = i + 1

    while (pos < len(data)):
        if mask[pos]:
            return pos, data[pos]
        pos += 1

    return len(data), default_value


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


def coeffs(x, y):
    if len(x) != len(y) or len(x) != 3:
        raise ValueError("Quadratic interpolation requires 3 points")
    return np.linalg.solve(
        np.array([[x[0]**2, x[0], 1], [x[1]**2, x[1], 1], [x[2]**2, x[2], 1]]),
        np.array([y[0], y[1], y[2]]))


def quadratic(data, mask):
    x = 0
    (x0, y0), (x1, y1) = (-1, 0), find_point_right(x, data, mask)
    if (x1 == len(data)):
        #No data points found! This is expected in some cases. Perform no interpolation
        return data, mask

    (x2, y2) = find_point_right(x1, data, mask)
    a, b, c = coeffs([x0, x1, x2], [y0, y1, y2])

    while x < len(data):
        #reached the end of interp. range, start a new one
        if (x == x2):
            (x0, y0) = (x2, y2)
            (x1, y1) = find_point_right(x0, data, mask)
            (x2, y2) = find_point_right(x1, data, mask)
            #edge case (literally). Act like there's a further point beyond the end of image with value 0
            if (x1 == x2):
                x2 += (x1 - x0)
            a, b, c = coeffs([x0, x1, x2], [y0, y1, y2])
        elif (mask[x] != 1):
            # Interpolate
            data[x] = max(min(a * x**2 + b * x + c, 255),
                          0)  #why is there no clamp function?
            # Mark point as containing value, so it can be used in next stage of 2D interpolation
            mask[x] = 1
        x += 1
    return data, mask


def coeffs_cubic(x, y):
    if len(x) != len(y) or len(x) != 4:
        raise ValueError("Cubic interpolation requires 4 points")
    return np.linalg.solve(
        np.array([[x[0]**3, x[0]**2, x[0], 1], [x[1]**3, x[1]**2, x[1], 1],
                  [x[2]**3, x[2]**2, x[2], 1], [x[3]**3, x[3]**2, x[3], 1]]),
        np.array([y[0], y[1], y[2], y[3]]))


#https://home.agh.edu.pl/~zak/downloads/MN3-2012.pdf slide 12
def cubic(data, mask):
    x = 0
    (x0, y0), (x1, y1) = (-1, 0), find_point_right(x, data, mask)
    if (x1 == len(data)):
        #No data points found! This is expected in some cases. Perform no interpolation
        return data, mask

    (x2, y2) = find_point_right(x1, data, mask)
    (x3, y3) = find_point_right(x2, data, mask)

    a, b, c, d = coeffs_cubic([x0, x1, x2, x3], [y0, y1, y2, y3])

    while x < len(data):
        if (x == x3):
            (x0, y0) = (x3, y3)
            (x1, y1) = find_point_right(x0, data, mask)
            (x2, y2) = find_point_right(x1, data, mask)
            (x3, y3) = find_point_right(x2, data, mask)

            #edge case (literally). Act like there's further points beyond the end of image with value 0
            if (x2 <= x1):
                x2 = x1 + (x1 - x0)
            if (x3 <= x2):
                x3 = x2 + (x2 - x1)

            a, b, c, d = coeffs_cubic([x0, x1, x2, x3], [y0, y1, y2, y3])
        elif (mask[x] != 1):
            #interpolate
            data[x] = max(min(a * x**3 + b * x**2 + c * x + d, 255), 0)
            mask[x] = 1
        x += 1

    return data, mask


def interpolation_2d(mosaic, mask, interp_f):
    output = np.zeros(mosaic.shape, dtype=np.uint8)
    h, w, _ = mosaic.shape

    #Perform interpolation on each color channel
    for channel in range(3):
        #Interpolate by rows
        print(f"Channel {channel}, interpolating rows...")
        for row in range(h):
            interp_row, interp_mask = interp_f(mosaic[row, :, channel],
                                               mask[row, :, channel])
            output[row, :, channel] = interp_row
            mask[row, :, channel] = interp_mask

        #Then by columns. Note we're using previous stage's output as input here
        print(f"Channel {channel}, interpolating columns...")
        for column in range(w):
            interp_col, _mask = interp_f(output[:, column, channel],
                                         mask[:, column, channel])
            output[:, column, channel] = interp_col

    return output


def mse_img(img1, img2):
    mses = [0, 0, 0]
    for c in range(3):
        mses[c] = np.square(img1[:, :, c] - img2[:, :, c]).mean()

    return mses


def mae_img(img1, img2):
    mses = [0, 0, 0]
    for c in range(3):
        mses[c] = np.abs(img1[:, :, c] - img2[:, :, c]).mean()

    return mses


def save_img(array, name):
    image = Image.fromarray(array)
    image.save(name)


pic = Image.open("lenna.png")
original = np.array(pic, dtype=np.uint8)

mask = make_mask(original.shape, window_bayer)
mosaic = make_mosaic(original, mask)

save_img(mask * 255, "mask.png")
save_img(mosaic, "mosaic.png")

interp_nn = interpolation_2d(mosaic, mask.copy(), nn)
interp_linear = interpolation_2d(mosaic, mask.copy(), linear)
interp_quadratic = interpolation_2d(mosaic, mask.copy(), quadratic)
interp_cubic = interpolation_2d(mosaic, mask.copy(), cubic)

results = {
    "nn": interp_nn,
    "linear": interp_linear,
    "quadratic": interp_quadratic,
    "cubic": interp_cubic
}

for key in results:
    save_img(results[key], key + ".png")
    print(key)
    print("MSE:", mse_img(original, results[key]))
    print("MAE:", mae_img(original, results[key]))
