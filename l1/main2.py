import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def boxcar(x, l = -0.5, r = 0.5):
    return (x > l) * (x <= r)

def triangular(x):
    return (1 - abs(x)) * boxcar(x, -1, 1)

def interpolate(x_r, x_dec, y_dec, kernel):
    y_vals = []
    for x in x_r:
        acc = 0
        for (p_x, p_y) in zip(x_dec, y_dec):
            acc += p_y * kernel(x - p_x)

        y_vals.append(acc)

    return y_vals

def coeffs(y1, y2, y3, x1, x2, x3):
    [a, b, c] = np.matmul(
        np.linalg.inv([[x1**2, x1, 1], [x2**2, x2, 1], [x3**2, x3, 1]]),
        [y1, y2, y3])
    return a, b, c

def quadratic(x_r, x_dec, y_dec):
    y_vals = [y_dec[0]]

    for i in range(1, len(x_dec) - 1, 2):
        x1, x2, x3 = x_dec[i - 1], x_dec[i], x_dec[i + 1]
        y1, y2, y3 = y_dec[i - 1], y_dec[i], y_dec[i + 1]
        a, b, c = coeffs(y1, y2, y3, x1, x2, x3)
        for x in x_r:
            if (x > x1) and (x < x3):
                y = a * (x**2) + b * x + c
                y_vals.append(y)
            elif (x == x3):
                y_vals.append(y3)

    return y_vals

def mse(y_real, y_pred):
    return  np.mean(np.square(np.subtract(y_real, y_pred)))

def mae(y_real, y_pred):
    return np.mean(np.abs(np.subtract(y_real, y_pred)))

def compare_interps(kind, t, y, x_remap, x_dec, y_dec):
    print(f"\n{kind} interpolation:")

    start = time.time()
    if (kind == "nearest"):
        inter_y = interpolate(x_remap, x_dec, y_dec, boxcar)
    elif (kind == "linear"):
        inter_y = interpolate(x_remap, x_dec, y_dec, triangular)
    elif (kind == "quadratic"):
        inter_y = quadratic(x_remap, x_dec, y_dec)
    else:
        print("invalid interpolation kind!")
        return
    end = time.time()

    sci_start = time.time()
    f = interp1d(x_dec, y_dec, kind=kind)
    sci_y = f(x_remap)
    sci_end = time.time()

    mse_our = mse(y, inter_y)
    mse_sci = mse(y, sci_y)

    mae_our = mae(y, inter_y)
    mae_sci = mae(y, sci_y)

    print(f"MSE our: {mse_our} | MSE Scipy: {mse_sci}")
    print(f"MAE our: {mae_our} | MAE Scipy: {mae_sci}")
    print(f"Our alg took: {end-start}s, Scipy took: {sci_end-sci_start}s")

    plt.clf()
    plt.plot(t, y, 'b')
    plt.plot(t, inter_y, 'r')
    plt.plot(t, sci_y, 'g')
    plt.savefig(f"{kind}.png")

def main():
    dec_factor = 10

    t = np.linspace(0, 10, 1001)
    y = [t[i]**2 + 3 * t[i] + 1 for i in range(len(t))]
    x_remap = [x / dec_factor for x in range(0, len(t))]
    x_dec = x_remap[::10]
    y_dec = y[::10]

    compare_interps("nearest", t, y, x_remap, x_dec, y_dec)
    compare_interps("linear", t, y, x_remap, x_dec, y_dec)
    compare_interps("quadratic", t, y, x_remap, x_dec, y_dec)


if __name__ == '__main__':
    main()