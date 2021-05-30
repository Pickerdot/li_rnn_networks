import numpy as np
import function
import random

"""
RNN化バージョン0509作成s-PFCのための層内結合ありのネットワーク
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
        self.dsr_prev = None
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

    def forward(self, v, si, sr, W, Wr, b, mode="None"):
        noise = self.random_del * (random.random()-0.5)*10
        # 変更点:層内結合の追加
        RI = -noise + np.dot(si, W)
        #print("RI:", RI.shape)
        #print("sr:", sr.shape)
        #print("Wr", Wr.shape)
        RI += np.dot(sr, Wr)
        v = self.function.eular_V(v, RI)

        s = self.function.sigmoid(v)
        if mode == "RI":
            return v, s, RI
        return v, s
