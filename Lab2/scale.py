from math import ceil, floor
import numpy as np
import sys
from PIL import Image

def get(arr2d, y, x, c):
    sy, sx, _ = arr2d.shape
    if (y >= sy or y < 0 or x >= sx or x < 0):
        return 0

    return arr2d[y, x, c]


def interpolate_lin(input, scale_factor, output):
    sy, sx, _ = input.shape
    for ny, nx, nc in np.ndindex(output.shape):
        y, x, _ = ny / scale_factor, nx / scale_factor, nc
        if y >= sy or x >= sx:
            continue
        x1, x2 = floor(x), ceil(x)
        y1, y2 = floor(y), ceil(y)

        q11 = get(input, y1, x1, nc)
        q12 = get(input, y2, x1, nc)
        q21 = get(input, y1, x2, nc)
        q22 = get(input, y2, x2, nc)

        if (x1 == x2 and y1 == y2):
            output[ny, nx, nc] = q11
        elif (x1 == x2):
            output[ny, nx, nc] = (y - y1) * q11 + (y2 - y) * q12
        elif (y1 == y2):
            output[ny, nx, nc] = (x - x1) * q11 + (x2 - x) * q21
        else:
            x_coeffs = np.array([x2 - x, x - x1], dtype=np.float32)
            samples = np.array([[q11, q12], [q21, q22]], dtype=np.float32)
            y_coeffs = np.array([[y2 - y], [y - y1]], dtype=np.float32)

            value = 1 / ((x2 - x1) * (y2 - y1)) * x_coeffs @ samples @ y_coeffs
            output[ny, nx, nc] = value


def interpolate(input, scale_factor, output):
    sy, sx, _ = input.shape
    for ny, nx, nc in np.ndindex(output.shape):
        y, x, c = round(ny / scale_factor), round(nx / scale_factor), nc
        if y >= sy or x >= sx:
            continue
        output[ny, nx, nc] = input[y, x, c]


def scale(input, scale_factor):
    sy, sx, c_count = input.shape
    output = np.zeros((int(np.ceil(
        sy * scale_factor)), int(np.ceil(sx * scale_factor)), c_count),
                      dtype=np.uint8)

    if (len(sys.argv) > 3 and sys.argv[3] == "nn"):
        interpolate(input, scale_factor, output)
    else:
        interpolate_lin(input, scale_factor, output)
    return output

def main():
    fname = sys.argv[1]
    scale_factor = float(sys.argv[2])
    pic = Image.open(fname)
    img = np.array(pic, dtype=np.uint8)

    output = scale(img, scale_factor)
    output_img = Image.fromarray(output)
    output_img.save("scaled.png")

    output = scale(output, 1 / scale_factor)
    output_img = Image.fromarray(output)
    output_img.save("reverted.png")
    


if __name__ == "__main__":
    main()