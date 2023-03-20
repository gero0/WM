import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def boxcar(x):
    if (x > -1 / 2 and x <= 1 / 2):
        return 1

    return 0


def triangular(x):
    if abs(x) < 1:
        return 1 - abs(x)
    else:
        return 0


def interpolate(x_r, x_dec, y_dec, kernel):
    y_vals = []
    for x in x_r:
        acc = 0
        for (p_x, p_y) in zip(x_dec, y_dec):
            acc += p_y * kernel(x - p_x)

        y_vals.append(acc)

    return y_vals


def mse(y_real, y_pred):
    acc = 0
    for (y1, y2) in zip(y_real, y_pred):
        acc += (y1 - y2)**2

    return acc / len(y_real)


def mae(y_real, y_pred):
    acc = 0
    for (y1, y2) in zip(y_real, y_pred):
        acc += abs(y2 - y1)

    return acc / len(y_real)


def main():
    dec_factor = 10

    t = np.linspace(0, 10, 1001)
    y = [t[i]**2 + 3 * t[i] + 1 for i in range(len(t))]
    x_remap = [x / dec_factor for x in range(0, len(t))]
    x_dec = x_remap[::10]
    y_dec = y[::10]


    start = time.time()
    inter_y = interpolate(x_remap, x_dec, y_dec, triangular)
    end = time.time()

    sci_start = time.time()
    f = interp1d(x_dec, y_dec, kind='linear')
    sci_y = f(x_remap)
    sci_end = time.time()

    mse_our = mse(y, inter_y)
    mse_sci = mse(y, sci_y)

    mae_our = mae(y, inter_y)
    mae_sci = mae(y, sci_y)

    print(f"MSE our: {mse_our} | MSE Scipy: {mse_sci}")
    print(f"MAE our: {mae_our} | MAE Scipy: {mae_sci}")
    print(f"Our alg took: {end-start}s, Scipy took: {sci_end-sci_start}s")

    plt.plot(t, y, 'b')
    plt.plot(t, inter_y, 'r')
    plt.plot(t, sci_y, 'g')
    plt.show()


if __name__ == '__main__':
    main()