
from cProfile import label
from copy import copy
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim([0, 10])
ax.set_ylim([0, 10])


coords = []

w0 = np.random.rand(1)
w1 = np.random.rand(1)

# power of polynom
power = 5
W = np.random.rand(power, 1)
W1 = copy(W)

print(W)
print(W[1:])
print(W[1][0])


def onclick(event):
    global w0, w1
    global W, W1
    print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          (event.button, event.x, event.y, event.xdata, event.ydata))

    coords.append((event.xdata, event.ydata))
    coords.sort(key=lambda tup: tup[0])
    x, y = list(map(list, zip(*coords)))

    plt.plot(x, y, 'o', color="red")

    w0, w1 = optim(w0, w1, x, y)
    prediction = model(w0, w1, x)
    ln, = plt.plot(x, prediction, color='blue', label='power: 1')

    W = optim1(W, x, y, regularization=False)
    prediction_generalized = model1(W, x)
    print(f'PRED{prediction_generalized[0]}')
    ln_g, = plt.plot(x, prediction_generalized[0], color='green', label=f'power: {power-1}, without L2')

    W1 = optim1(W1, x, y, regularization=True)
    pred_with_regularization = model1(W1, x)
    ln_g_reg, = plt.plot(x, pred_with_regularization[0], color='yellow',label=f'power: {power-1}, with L2')
    plt.legend()

    fig.canvas.draw()
    ln.remove()
    ln_g.remove()
    ln_g_reg.remove()


def mypower(x, power):
    return np.array([a**power for a in x])


def get_matrix(x, W, powers):
    result = []
    for i in powers:
        result.extend(mypower(x, i))
    result = np.array(result).reshape(len(W)-1, len(x))
    print(f'TEMP\n{result}')
    return result


def model1(W, x):
    x_matrix = get_matrix(x, W, range(1, len(W)))
    print(f'MATRIX\n{x_matrix}')
    print(x_matrix.shape)
    print(W[1:].shape)
    result = W[1:].T @ x_matrix
    print(f'RESULT\n{result}')
    result[:] += W[0]
    print(W[0])
    print(f'RESULT with w0\n{result}')
    return result


def loss1(W, x, y):
    pred = model1(W, x)
    return (1/len(x))*(np.square(pred - y)).sum()


def optim1(W, x, y, lmbda=3.e5, regularization=False, Lr=5.e-8):
    pred = model1(W, x)
    dW = [2/len(x) * (((pred - y)*mypower(x, i)).sum()) for i in range(len(W))]
    if regularization:
        for i in range(len(W)):
            dW[i] += 2 * lmbda * W[i]
    for i in range(len(dW)):
        W[i][0] = W[i][0] - Lr * dW[i]

    return W


def model(w0, w1, x):
    return w0 + w1*x


def loss(w0, w1, x, y):
    pred = model(w0, w1, x)
    return (1/len(x))*(np.square(pred - y)).sum()


def optim(w0, w1, x, y, Lr=0.01):
    pred = model(w0, w1, x)

    dw1 = 2/len(x) * (((pred-y)*x).sum())
    dw0 = 2/len(x) * ((pred-y).sum())

    w0 = w0 - Lr*dw0
    w1 = w1 - Lr*dw1
    return w0, w1


def iter(w0, w1, x, y, t=5):
    for i in range(t):
        w0, w1 = optim(w0, w1, x, y)
    return w0, w1


cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
