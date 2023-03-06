from PIL import Image
import numpy as np
from math import pi, sin, cos
import pathlib

# maciej.filinski.staff.iiar.pwr.edu.pl

M = 64
N = 2048
IMG_SIZE = 800


def polar_to_cart(r, ang):
    x = r * cos(ang)
    y = r * sin(ang)
    return x, y


def draw_prop(path, i, m, img_size):
    x = np.linspace(0, 2 * pi, N)
    fx = np.sin(3 * x + (m * pi) / 10)

    points = [polar_to_cart(r, ang) for ang, r in zip(x, fx)]
    points = [((x + 1) / 2, (y + 1) / 2) for x, y in points]
    points = [(int(x * img_size), img_size - int(y * img_size)) for x, y in points]

    img = Image.new(mode="RGB", size=(img_size + 1, img_size + 1))

    for point in points:
        img.putpixel(point, (255, 255, 255))
    print(f"{path}/prop{i}.png")
    img.save(f"{path}/prop{i}.png")


ms = np.linspace(-M / 2, M / 2, 64)

pathlib.Path("images").mkdir(exist_ok=True)

for i, m in enumerate(ms):
    draw_prop("images", i, m, IMG_SIZE)
