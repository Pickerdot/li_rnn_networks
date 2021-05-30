import numpy as np
import rnn_neuron as nu
import function

# f_FPC_unit: a one unit of f-PFC,2

W = np.array([[0, 1], [0.917, 0]])


class rnn_layer_unit:
    def __init__(self):
        self.random_del = 1
        self.dt = 0.01
        self.Tau = 10
        self.W = None
        self.Wr = None
        self.b = None
        self.V = None
        self.S = None
        self.one_layer = nu.neuron()
        self.one_layer.def_parameter(
            self.Tau, self.dt, self.random_del)
        self.function = function.function()
        self.function.def_parameter(
            self.Tau, self.dt, self.random_del)
        self.mode = None

    def def_parameter(self, Tau, dt, random_del):
        self.random_del = random_del
        self.dt = dt
        self.Tau = Tau
        self.one_layer.def_parameter(
            self.Tau, self.dt, self.random_del)
        # print("**f=pfc")
        # print(self.random_del)

    def initiate(self, v, b, w, wr, mode=None):
        self.W = w
        self.Wr = wr
        self.V = v
        self.S = self.function.sigmoid(v)
        self.b = b
        self.mode = mode

    def forward(self, si):
        self.V, self.S = self.one_layer.forward(
            self.V, si, self.S, self.W, self.Wr, self.b, self.mode)
        return self.V, self.S
