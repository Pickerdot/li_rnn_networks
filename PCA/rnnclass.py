from optimizer import optimizer_SGD, AdaGrad, NormGrad, SGD
import numpy as np
from functions import sigmoid, sigmoid_back, clip_grads


class Loss:
    def __init__(self):
        self.Loss = None
        self.dout = None

    def forward(self, out, t):
        self.Loss = 1/2 * np.sum((out - t)**2)
        self.dout = out - t
        return self.Loss

    def backward(self):
        return self.dout


class RNNneuron:
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

    def forward(self, x, h_prev):
        # クラスの初期化時に格納した重みとバイアスの取り出し
        W, Wh, b = self.params
        # yはニューロン内部の値
        #f = open("E:\研究一時ファイル\BP\TEST_1120\Fh.txt", mode="a")
        if h_prev is None:
            y = np.dot(x, W) + b
        else:
            y = np.dot(h_prev, Wh) + np.dot(x, W) + b

        #w = "\nWh:" + str(Wh) + "\nh_prev:" + str(h_prev) + "\n:" + str(y)

        # f.write(w)

        # Zが出力
        z = sigmoid(y)
        self.h_prev = z
        self.F_container = [W, Wh, b, x, y, z]
        return z, self.F_container

    def backward(self, dz, h_prev):
        #f = open("E:\研究一時ファイル\BP\TEST_1120\Wh.txt", mode="a")
        W, Wh, b, x, y, z = self.F_container
        dh_prev = self.dh_prev
        # 過去時刻からの勾配の合算
        if dh_prev is None:
            dz = dz
        else:
            dz = dh_prev + dz
        # 出力部の逆伝搬（シグモイド版）
        dy = sigmoid_back(z, dz)
        db = dy
        dW = np.dot(x.T, dy)
        dx = np.dot(dy, W.T)
        dWh = np.dot(h_prev.T, dy)
        dh_prev = np.dot(dy, Wh.T)

        #w = "\ndWh:" + str(dWh) + "\nh_prev:" + str(h_prev) + "\ndy:" + str(dy)

        # f.write(w)

        # 勾配クリッピングの実行
        self.drads, self.clipper = clip_grads(self.grads, self.NormGrad)

        self.dh_prev = dh_prev

        # self.gradsに更新に行かう勾配を格納
        self.grads[0][...] = dW
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        # オプティマイザーによりself.paramsの値を更新
        # self.params = optimizer_SGD(self.lr, self.params, self.grads)
        self.params = self.optimizer.update(self.params, self.grads)
        # すべての結果をself.containerに格納
        self.container = [dy, db, dW, dWh, dx]

        # f.close

        return dx, self.container

    def setlr(self, lr, model=0):
        self.lr = lr
        if model == 0:
            self.optimizer = SGD(self.lr)
        elif model == 1:
            self.optimizer = AdaGrad(self.lr)
        elif model == 2:
            self.optimizer = NormGrad(self.lr)

    def viewlr(self):
        return self.optimizer.viewlr()

    def change_lr(self, New_lr):
        self.optimizer.change_lr(New_lr)

    def reset(self):
        self.h_prev = None
        self.dh_prev = None

    def clipper_Chech(self):
        return self.clipper

    def change_NormGrad(self, NormGrad):
        # 勾配クリッピングのしきい値の変更
        self.NormGrad = NormGrad


class BPneuron:
    def __init__(self, W, b):
        # 引数として受けた重みとバイアスをself.aramsに格納
        self.params = [W, b]
        # 更新前に勾配をまとめてオプティマイザーに送るための入れ物（中身はparamsに対応している必要あり）
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        # クラス外へ中身を持っていくための入れ物
        self.container = np.empty(0)
        # 学習率の格納
        self.lr = 0.01
        self.optimizer = AdaGrad(self.lr)

    def forward(self, x):
        # クラスの初期化時に格納した重みとバイアスの取り出し
        W, b = self.params
        # yはニューロン内部の値
        y = np.dot(x, W)+b
        # Zが出力
        z = sigmoid(y)
        self.container = [W, b, x, y, z]
        return z, self.container

    def backward(self, dz):
        W, b, x, y, z = self.container
        # 出力部の逆伝搬（シグモイド版）
        dy = sigmoid_back(z, dz)
        db = dy
        dW = np.dot(x.T, dy)
        dx = np.dot(dy, W.T)

        # self.gradsに更新に行かう勾配を格納
        self.grads[0][...] = dW
        self.grads[1][...] = db

        # オプティマイザーによりself.paramsの値を更新
        # self.params = optimizer_SGD(self.lr, self.params, self.grads)
        self.params = self.optimizer.update(self.params, self.grads)
        # すべての結果をself.containerに格納
        self.container = [dy, db, dW, dx]

        return dx, self.container

    def setlr(self, lr, model=0):
        self.lr = lr
        if model == 0:
            self.optimizer = SGD(self.lr)
        elif model == 1:
            self.optimizer = AdaGrad(self.lr)
        elif model == 2:
            self.optimizer = NormGrad(self.lr)

    def viewlr(self):
        return self.optimizer.viewlr()

    def change_lr(self, New_lr):
        self.optimizer.change_lr(New_lr)
