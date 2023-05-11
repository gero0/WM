import numpy as np
import sys
from PIL import Image


def deg_to_rad(alpha):
    return alpha * np.pi / 180


def main():
    fname = sys.argv[1]
    times = int(sys.argv[3])

    alpha = deg_to_rad(float(sys.argv[2]))
    pic = Image.open(fname)
    img = np.array(pic, dtype=np.uint8)
    rMat = np.array([[np.cos(alpha), -np.sin(alpha)],
                     [np.sin(alpha), np.cos(alpha)]])

    border = max(img.shape[0] // 2, img.shape[1] // 2)
    buffer_shape = (img.shape[0] + border, img.shape[1] + border, 3)
    input = np.zeros(buffer_shape, dtype=np.uint8)
    half = border // 2
    input[half:half + img.shape[0], half:half + img.shape[1]] = img

    for _ in range(times):
        output = np.zeros(buffer_shape, dtype=np.uint8)
        for y, x, c in np.ndindex(buffer_shape):
            c_y = y - input.shape[0] // 2
            c_x = x - input.shape[1] // 2
            orig_y, orig_x = rMat @ [c_y, c_x]
            orig_y, orig_x = int(np.round(orig_y)), int(np.round(orig_x))
            orig_y, orig_x = (orig_y + input.shape[0] // 2,
                              orig_x + input.shape[1] // 2)
            if (orig_y > 0 and orig_y < input.shape[0] and orig_x > 0
                    and orig_x < input.shape[1]):
                output[y, x, c] = input[orig_y, orig_x, c]

        input = output

    output = output[half:half + img.shape[0], half:half + img.shape[1]]
    output = Image.fromarray(output)
    output.save("rotated.png")


if __name__ == "__main__":
    main()