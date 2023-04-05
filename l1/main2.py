import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def boxcar(x, l = -0.5, r = 0.5):
    return (x >= l) * (x < r)

def triangular(x):
    return (1 - abs(x)) * boxcar(x, -1, 1)

def keys(x):
    '''Keys' cubic interpolation function, blatantly stolen from
    https://en.wikipedia.org/wiki/Bicubic_interpolation
    and https://www.ncorr.com/download/publications/keysbicubic.pdf'''
    return ((1.5 * abs(x)**3 - 2.5 * abs(x)**2 + 1) * boxcar(abs(x), 0, 1)
          + (-0.5 * abs(x)**3 + 2.5 * abs(x)**2 - 4 * abs(x) + 2) * boxcar(abs(x), 1, 2))

def interpolate(x_r, x_dec, y_dec, kernel):
    y_vals = []
    for x in x_r:
        acc = 0
        for (p_x, p_y) in zip(x_dec, y_dec):
            acc += p_y * kernel(x - p_x)

        y_vals.append(acc)
    return y_vals

def coeffs(x, y):
    if len(x) != len(y) or len(x) != 3: raise ValueError("Quadratic interpolation requires 3 points")
    return np.linalg.solve(np.array([[x[0]**2, x[0], 1], [x[1]**2, x[1], 1], [x[2]**2, x[2], 1]]), np.array([y[0], y[1], y[2]]))

def quadratic(x_r, x_dec, y_dec):
    y_vals = [y_dec[0]]

    for i in range(1, len(x_dec) - 1, 2):
        xs = [x_dec[i - 1], x_dec[i], x_dec[i + 1]]
        ys = [y_dec[i - 1], y_dec[i], y_dec[i + 1]]
        a, b, c = coeffs(xs, ys)
        for x in x_r:
            if (x > xs[0]) and (x < xs[2]):
                y = a * x**2 + b * x + c
                y_vals.append(y)
            elif (x == xs[2]):
                y_vals.append(ys[2])

    return y_vals

def cubic_interp(x_r, x_dec, y_dec):
    y_vals = []

    #distance between two x'es - assuming points are sampled at equal intervals
    dist = x_dec[1] - x_dec[0]

    def c(k):
        if(k==-1):
            return (3 * y_dec[0] - 3 * y_dec[1] + y_dec[2])
        if(k == len(x_dec)):
            return (3 * y_dec[-1] - 3 * y_dec[-2] + y_dec[-3])
        
        return y_dec[k]
    
    def x_k(k):
        if(k == -1):
            return x_dec[0] - dist
        if(k == len(x_dec)):
            return x_dec[-1] + dist

        return x_dec[k]
    
    """
    https://www.ncorr.com/download/publications/keysbicubic.pdf
    According to the paper we need to convolve from k=-1 to k=N+1.
    The paper also says we know samples from range k=0 to k=N, while here.
    we only know range k=0 to k=N-1, that's why N+1 is excluded in our calculations.
    Idk why but it works.
    c_k and x_k are helper functions that define values for x and y outside of known range.
    For x we're just adding/subtracting interval between points (it must be constant between points for this to make sense).
    The y values are calculated based on equations from the paper.
    The rest is just ye olde convolution.
    """
    for x in x_r:
        acc = 0
        for k in range(-1, len(x_dec) + 1):
            acc += c(k) * keys(x - x_k(k))

        y_vals.append(acc)
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
    elif (kind == "cubic"):
        inter_y = cubic_interp(x_remap, x_dec, y_dec)
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

    t = np.linspace(0, 100, 1001)
    y = t**2 + 3 * t + 1
    x_remap = [x / dec_factor for x in range(0, len(t))]
    x_dec = x_remap[::10]
    y_dec = y[::10]

    compare_interps("nearest", t, y, x_remap, x_dec, y_dec)
    compare_interps("linear", t, y, x_remap, x_dec, y_dec)
    compare_interps("quadratic", t, y, x_remap, x_dec, y_dec)
    compare_interps("cubic", t, y, x_remap, x_dec, y_dec)


if __name__ == '__main__':
    main()