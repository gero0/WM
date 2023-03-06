from PIL import Image
import numpy as np
import pathlib

# maciej.filinski.staff.iiar.pwr.edu.pl

M = 64
N = 2048
IMG_SIZE = 512
Ms = np.linspace(-M / 2, M / 2, 64)


def polar_to_cart(r, ang):
    x = r * np.cos(ang)
    y = r * np.sin(ang)
    return x, y


def draw_prop(path, n_wings, i, m, img_size):
    x = np.linspace(0, 2 * np.pi, N)
    fx = np.sin(n_wings * x + (m * np.pi) / 10)

    points = [polar_to_cart(r, ang) for ang, r in zip(x, fx)]
    points = [((x + 1) / 2, (y + 1) / 2) for x, y in points]
    points = [
        (int(x * img_size - 1), img_size - 1 - int(y * img_size - 1)) for x, y in points
    ]

    img = Image.new(mode="RGB", size=(img_size, img_size))

    for point in points:
        img.putpixel(point, (255, 255, 255))
    print(f"{path}/prop{i}.png")
    img.save(f"{path}/prop{i}.png")


def main():
    pathlib.Path("images").mkdir(exist_ok=True)
    for i, m in enumerate(Ms):
        draw_prop("images", 3, i, m, IMG_SIZE)


if __name__ == "__main__":
    main()
