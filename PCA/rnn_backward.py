import numpy as np
import rnn_neuron as nu
import function


class layer_backword:
    def __init__(self, W, Wh, b):
        # 引数として受けた重みとバイアスをself.aramsに格納
        self.params = [W, Wh, b]
        # 更新前に勾配をまとめてオプティマイザーに送るための入れ物（中身はparamsに対応している必要あり）
        self.grads = [np.zeros_like(W), np.zeros_like(Wh), np.zeros_like(b)]
        # クラス外へ中身を持っていくための入れ物
        self.F_container = np.empty(0)
        self.B_container = np.empty(0)
        # RNN層の中身の入れ物
        self.dh_prev = None
        # 学習率の格納
        self.lr = 0.01
        # オプティマイザーの定義（初期値SGD）
        self.optimizer = SGD(self.lr)
        # クリッピングの実行フラグ
        self.clipper = 0
        # 勾配クリッピングのしきい値（初期値0.02)
        self.NormGrad = 0.02

    def backward(self, ):
