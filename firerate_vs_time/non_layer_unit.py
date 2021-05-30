import numpy as np
import non_neuron as nu
import function
import random

# f_FPC_unit: a one unit of f-PFC,2
# rmm_layer_unitの複製0519

W = np.array([[0, 1], [0.917, 0]])


class non_layer_unit:
    def __init__(self):
        self.random_del = 1
        self.dt = 0.01
        self.Tau = 10
        self.W = None
        self.b = None
        self.V = None
        self.S = None
        self.si = None
        self.one_layer = nu.neuron()
        self.one_layer.def_parameter(
            self.Tau, self.dt, self.random_del)
        self.function = function.function()
        self.function.def_parameter(
            self.Tau, self.dt, self.random_del)
        self.mode = None

        self.Loss = None
        self.dout = None

        self.lr = 0.01

    def set_lr(self, lr):
        self.lr = lr

    def loss(self, t):
        #print("S:", self.S)
        #print("t:", t)
        self.Loss = 1/2 * np.sum((self.S - t)**2)
        #print("Loss", self.Loss)

    def def_parameter(self, Tau, dt, random_del):
        self.random_del = random_del
        self.dt = dt
        self.Tau = Tau
        self.one_layer.def_parameter(
            self.Tau, self.dt, self.random_del)
        # print("**f=pfc")
        # print(self.random_del)

    def initiate(self, v, b, w, mode=None):
        self.W = w
        self.V = v
        self.S = self.function.sigmoid(v)
        self.b = b
        self.mode = mode

    def sgd(self, dW, db):
        #print("W1", self.W[0])
        self.W -= self.lr * dW
        self.b -= self.lr * db
        #print("W2", self.W[0])
        #print("lr", self.lr)

    def forward(self, si):
        self.si = si
        self.V, self.S = self.one_layer.forward(
            self.V, si, self.W, self.b)
        #print("S", self.S)
        return self.V, self.S

    def forward_dry(self, si):
        self.si = si
        self.V, self.S = self.one_layer.forward_dry(
            self.V, si, self.W, self.b)
        #print("S*", self.S)
        return self.V, self.S

    def backward(self, t):
        dy = self.function.sigmoid_back(self.S, self.Loss)
        dW = np.dot(self.si.T, dy)
        dx = np.dot(dy, self.W.T)
        self.sgd(dW, dy)

    def output_loss(self, t):
        self.dout = self.S - t
        return self.dout[0][0]

    def printS(self):
        print(self.S)
