import math
import numpy as np
import random

# 基本関数

"""
****************************************
 基本パラメーター
 必ず上層でdef\parameterで定義すること
"""

# ここまで


class function:
    def __int__(self):
        self.Tau = 10
        self.dt = 0.01
        self.random_del = 1

    def def_parameter(self, Tau, dt, random_del):
        self.Tau = Tau
        self.dt = dt
        self.random_del = random_del
        # print("**func")
        # print(self.random_del)

    def sigmoid(self, x):
        y = 1/(1 + np.exp(-5*(x-0.5)))
        return y

    def f_V(self, x, RI):
        sum = 0
        sum = -x + RI
        return (sum / self.Tau)
        print(RI)

    def eular_V(self, x, i):
        x += self.dt * self.f_V(x, i)
        return x

    def noise(self):
        noise = self.random_del * (random.random()-0.5)
        return noise
