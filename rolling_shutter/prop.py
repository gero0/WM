from PIL import Image
import numpy as np
import pathlib

M = 64
N = 2048
IMG_SIZE = 256

#todo: make gif/vid

def main():
    camera_l = 16
    generate_frames(3, -M / 2, "frames", IMG_SIZE, 16, int(256 / camera_l))


def generate_frames(n_wings, m, path, img_size, camera_l, frames_to_render):
    m_diff = ((M / 2) - (-M / 2)) / 64
    pathlib.Path(path).mkdir(exist_ok=True)
    x = np.linspace(0, 2 * np.pi, N)

    for i in range(frames_to_render):
        img = Image.new(mode="RGB", size=(img_size, img_size))
        for strip in range(0, int(img_size / camera_l)):
            fx = np.sin(n_wings * x + (m * np.pi) / 10)
            m += m_diff
            points = get_raster_points(x, fx, img_size)
            for point in points:
                y = point[1]
                if y >= strip * camera_l and y < (strip + 1) * camera_l:
                    img.putpixel(point, (255, 255, 255))

        img.save(f"{path}/prop{i}.png")


def get_raster_points(x, fx, img_size):
    points = [polar_to_cart(r, ang) for ang, r in zip(x, fx)]
    points = [((x + 1) / 2, (y + 1) / 2) for x, y in points]
    points = [
        (int(x * img_size - 1), img_size - 1 - int(y * img_size - 1)) for x, y in points
    ]
    return points


def polar_to_cart(r, ang):
    x = r * np.cos(ang)
    y = r * np.sin(ang)
    return x, y


if __name__ == "__main__":
    main()
