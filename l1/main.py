import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

t = np.linspace(0, 10, 1000)
y = [t[i]**2+3*t[i]+1 for i in range(len(t))]
points = [(x, y) for (x, y) in zip(t,y)]
p_dec = [points[i] for i in range(len(points)) if (i+1)%10==0 or i==0]

def interpolate_nn(x, pair):
    d1 = abs(x - pair[0][0])
    d2 = abs(x - pair[1][0])

    if d1 < d2:
        return pair[0][1]
    else:
        return pair[1][1]

def interpolate_lin(x, pair):
    x1, y1 = pair[0]
    x2, y2 = pair[1]
    return (y2 - y1) / (x2 - x1) * (x - x1) + y1

def interpolate(inter_fun, inter_n):

    inter_x = []
    inter_y = []

    for i in range(0, len(p_dec) - 1):
        pair = p_dec[i], p_dec[i+1]
        inbetween_x = np.linspace(pair[0][0], pair[1][0], inter_n)

        for x in inbetween_x:
            y_int = inter_fun(x, pair)
            inter_x.append(x)
            inter_y.append(y_int)

    return (inter_x, inter_y)

def mse(y_real, y_pred):
    acc = 0
    for (y1, y2) in zip(y_real, y_pred):
        acc += (y1-y2)**2
    
    return acc / len(y_real)

def main():
    inter_x, inter_y = interpolate(interpolate_lin, 10)
    
    dec_x = [x for (x,_) in p_dec]
    f = sp.interp1d(dec_x, y, kind='linear')
    ynew = f(t)

    print(len(y), len(inter_y))
    print(mse(y, inter_y))
    print(mse(y, ynew))
    plt.figure()
    plt.plot(t, y)
    #plt.figure()
    plt.plot(inter_x, inter_y, 'r')
    plt.show()

if __name__ == '__main__':
    main()