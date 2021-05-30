import numpy as np
import function
import random

"""
RNN化バージョン0509作成s-PFCのための層内結合ありのネットワーク
rnn_neuronの複製0519
"""

# ニューロン単独
# このプログラムでは行列で定義されたニューロンの出力結果を返す
# ユニットの定義は上層で行う
# 引数として上から結合加重Wとバイアスbを受け取る
# forによる繰り返しは行わないあくまで一回のみ計算

"""
基本パラメーター
**************************
"""
Tau = 10
dt = 0.01
random_del = 0.5


class neuron:
    def __init__(self):
        self.random_del = 1
        self.dt = 0.01
        self.Tau = 10
        self.V = None
        self.S = None
        # 基本関数の定義と各パラメーターの設定
        self.function = function.function()
        self.function.def_parameter(
            self.Tau, self.dt, self.random_del)

    def def_parameter(self, Tau, dt, random_del):
        self.random_del = random_del
        self.dt = dt
        self.Tau = Tau
        self.function.def_parameter(
            self.Tau, self.dt, self.random_del)
        # print("**neuron")
        # print(self.random_del)

    def forward(self, v, si, W, b, mode="None"):
        noise = self.random_del * (random.random()-0.5)
        # 変更点:層内結合の追加
        RI = -noise + np.dot(si, W) + b
        v = self.function.eular_V(v, RI)
        s = self.function.sigmoid(v)
        if mode == "RI":
            return v, s, RI
        return v, s

    def forward_dry(self, v, si, W, b, mode="None"):
        noise = self.random_del * (random.random()-0.5)
        #print("si", si)
        # 変更点:層内結合の追加
        RI = -noise + np.dot(si, W) + b
        #print("RI:", RI)
        v += RI
        s = self.function.sigmoid(v)
        if mode == "RI":
            return v, s, RI
        return v, s

    def forward_it(self, v, s, W, b, input, mode="None"):
        noise = self.random_del * (random.random()-0.5)
        RI = -noise + np.dot(s, W) + input
        v = self.function.eular_V(v, RI)
        s = self.function.sigmoid(v)
        if mode == "RI":
            return v, s, RI
        return v, s
