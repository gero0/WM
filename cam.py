import pathlib
from PIL import Image

l = 2
K = 256
frames = 64
IMG_SIZE = 256

frames_to_gen = int(K / l)

pathlib.Path("rolling").mkdir(exist_ok=True)

for frame_i in range(0, frames_to_gen):
    frame = Image.new(mode="RGB", size=(IMG_SIZE, IMG_SIZE))
    for i in range(0, int(IMG_SIZE / l)):
        img_i = (frame_i + i) % frames
        img = Image.open(f"./images/prop{img_i}.png")
        region = img.crop((0, i * l, 256, (i + 1) * l))
        frame.paste(region, (0, i * l, 256, (i + 1) * l))

    frame.save(f"rolling/frame_{frame_i}.png")