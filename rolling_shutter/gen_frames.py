# from prop import generate_frames
from prop import get_raster_points
from PIL import Image
import numpy as np
import pathlib

M = 64
N = 2048
IMG_SIZE = 256

def generate_frames(n_wings, m, path, img_size, camera_l, frames_to_render):
    m_diff = ((M / 2) - (-M / 2)) / 64
    # pathlib.Path(path).mkdir(exist_ok=True)
    x = np.linspace(0, 2 * np.pi, N)
    frames = []

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
        frames.append(img)
    img.save(f"{path}/prop{camera_l}.gif", format="GIF", append_images=frames,
               save_all=True, duration=100, loop=0)
        
    

def main():
    pathlib.Path("frames").mkdir(exist_ok=True)
    for l in range(1,17):
        camera_l = l
        generate_frames(3, -M / 2, "frames", IMG_SIZE, l, int(256 / camera_l))
        
if __name__ == "__main__":
    main()