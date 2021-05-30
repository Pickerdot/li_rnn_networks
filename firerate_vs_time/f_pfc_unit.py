import math
import numpy as np
import random
import non_neuron as nu

# f_FPC_unit: a one unit of f-PFC,2

W = np.array([[0, 1], [0.917, 0]])


class f_PFC_unit:
    def __init__(self, V_d, V_c):
        self.random_del = 1
        self.dt = 0.01
        self.Tau = 10
        self.Wl = np.array([[0, 0.99], [0.922, 0]])
        self.Wh = np.array([[0, 1.02], [0.875, 0]])
        self.b = np.array([[0.0, 0.0]])
        self.V_d = V_d
        self.V_c = V_c
        self.f_FPC_unit_d = nu.neuron()
        self.f_FPC_unit_d.def_parameter(
            self.Tau, self.dt, self.random_del)
        self.f_FPC_unit_c = nu.neuron()
        self.f_FPC_unit_c.def_parameter(
            self.Tau, self.dt, self.random_del)

    def def_parameter(self, Tau, dt, random_del):
        self.random_del = random_del
        self.dt = dt
        self.Tau = Tau
        self.f_FPC_unit_c.def_parameter(
            self.Tau, self.dt, self.random_del)
        self.f_FPC_unit_d.def_parameter(
            self.Tau, self.dt, self.random_del)

    def forward(self, S_d, S_c, ITd, ITc):
        b_d = np.zeros_like(S_d)
        b_c = np.zeros_like(S_c)
        self.V_d, S_d = self.f_FPC_unit_d.forward_it(
            self.V_d, S_d, self.Wl, b_d, ITd)
        self.V_c, S_c = self.f_FPC_unit_d.forward_it(
            self.V_c, S_c, self.Wl, b_c, ITc)
        return S_d, S_c


"""
    def forward(self, S_d, S_c, ITd, ITc):
        b_d = np.zeros_like(S_d)
        b_c = np.zeros_like(S_c)
        self.V_d, S_d = self.f_FPC_unit_d.forward(
            self.V_d, S_d, self.Wl, b_d, ITd)
        self.V_c, S_c = self.f_FPC_unit_c.forward(
            self.V_c, S_c, self.Wl, b_c, ITc)
        return S_d, S_c
"""
