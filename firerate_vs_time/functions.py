# coding: utf-8
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x-1e-7))


def sigmoid_grad(z):
    return sigmoid(z) * (1 - sigmoid(z))


def sigmoid_back(z, dz):
    return dz*sigmoid(z) * (1 - sigmoid(z))


def tanh(x):
    y = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    return y


def tanh_grad(x):
    y = x*(1-x**2)
    return y

# 勾配クリッピング


def clip_grads(grads, max_norm=0.25):
    done = 0
    total_norm = 0
    for i in range(len(grads)):
        total_norm += np.sum(grads[i] ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        done = 1
        for i in range(len(grads)):
            grads[i] *= (rate+1)

    return grads, done
