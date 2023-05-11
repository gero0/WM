from PIL import Image
import numpy as np
import sys


def mse_img(img1, img2):
    mses = [0, 0, 0, 0]
    for c in range(3):
        mses[c] = np.square(img1[:, :, c] - img2[:, :, c]).mean()

    mses[3] = sum(mses[0:3])
    return mses


def mae_img(img1, img2):
    mses = [0, 0, 0, 0]
    for c in range(3):
        mses[c] = np.abs(img1[:, :, c] - img2[:, :, c]).mean()

    mses[3] = sum(mses[0:3])
    return mses


def main():
    file1, file2 = sys.argv[1], sys.argv[2]
    img1 = Image.open(file1)
    img2 = Image.open(file2)

    img1 = np.array(img1, dtype=np.uint8)
    img2 = np.array(img2, dtype=np.uint8)

    print(f"MSE is: {mse_img(img1, img2)}")
    print(f"MAE is: {mae_img(img1,img2)}")


if __name__ == "__main__":
    main()