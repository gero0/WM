import numpy as np
import sys
from PIL import Image


def deg_to_rad(alpha):
    return alpha * np.pi / 180


def main():
    fname = sys.argv[1]
    alpha = deg_to_rad(float(sys.argv[2]))
    pic = Image.open(fname)
    img = np.array(pic, dtype=np.uint8)
    rMat = np.array([[np.cos(alpha), -np.sin(alpha)],
                     [np.sin(alpha), np.cos(alpha)]])

    output = np.zeros(img.shape, dtype=np.uint8)
    for y, x, c in np.ndindex(output.shape):
        c_y = y - img.shape[0] // 2
        c_x = x - img.shape[1] // 2
        orig_y, orig_x = rMat @ [c_y, c_x]
        orig_y, orig_x = int(np.round(orig_y)), int(np.round(orig_x))
        orig_y, orig_x = (orig_y + img.shape[0] // 2, orig_x + img.shape[1] // 2)
        if (orig_y > 0 and orig_y < img.shape[0] and orig_x > 0
                and orig_x < img.shape[1]):
            output[y, x, c] = img[orig_y, orig_x, c]

    output = Image.fromarray(output)
    output.save("rotated.png")


if __name__ == "__main__":
    main()