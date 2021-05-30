import numpy as np
from random import random
import matplotlib.pyplot as plt
from IPython.display import clear_output
from rnn_layer_unit import rnn_layer_unit
from non_layer_unit import non_layer_unit


class test_network:
    def __init__(self, I, H, O, W01_size, W11_size, W12_size, sequence_length, w_path=None):
        self.O = O
        self.H = H
        self.I = I

        # 重みとバイアスの定義
        self.W01 = W01_size*np.random.rand(I, H)-W01_size/2
        self.W11 = W11_size*np.random.rand(H, H)-W11_size/2
        self.W12 = W12_size*np.random.rand(H, O)-W12_size/2
        self.V2 = np.zeros((1, H))
        self.V3 = np.zeros((1, O))
        self.b2 = np.zeros((1, H))
        self.b3 = np.zeros((1, O))

        # モデルの生成
        self.Seccond_layer = rnn_layer_unit()
        self.Third_layer = non_layer_unit()

        # １層、２層も初期化
        self.Seccond_layer.initiate(self.V2, self.b2, self.W01, self.W11)
        self.Third_layer.initiate(self.V3, self.b3, self.W12)

        self.loss_memo = []
        self.b_lr_memo = []
        self.dx_memo = np.empty(0)
        self.bo_memo = np.empty(0)
        self.dbo_memo = np.empty(0)
        self.W12_memo = np.empty(0)

        self.out1 = None
        self.out2 = None

        # realtime化に必要なパラメーター
        self.random_del = 1
        self.dt = 0.01
        self.Tau = 100
        self.Step_length = 100
        self.Seccond_layer.def_parameter(self.Tau, self.dt, self.random_del)
        self.Seccond_layer.def_parameter(self.Tau, self.dt, self.random_del)

        self.lr = 0.01

        # RNNの繰り返し期間sequence_lengthの格納
        self.sequence_length = sequence_length
        # RNNの時刻ごとの出力の入れ物
        self.h_prev = np.empty((sequence_length, H))

        # Abeloss用のlossの入れもの
        self.loss_container = 0

        # 学習率の減数率指数
        self.lr_index = 2

        # 誤差の最小範囲
        self.min_loss = 0.003

    def def_parameter(self, Tau, dt, random_del, Step_length):
        self.random_del = random_del
        self.dt = dt
        self.Tau = Tau
        self.Step_length = Step_length
        self.Seccond_layer.def_parameter(self.Tau, self.dt, self.random_del)
        self.Seccond_layer.def_parameter(self.Tau, self.dt, self.random_del)

    def set_Wandb(self, W01, W11, W12, bh, bo):
        self.W01 = W01
        self.W11 = W11
        self.W12 = W12
        self.b2 = bh
        self.b3 = bo
        self.Seccond_layer.initiate(self.V2, self.b2, self.W01, self.W11)
        self.Third_layer.initiate(self.V3, self.b3, self.W12)

    def network_reset(self):
        self.V2 = np.zeros((1, self.H))
        self.V3 = np.zeros((1, self.O))
        self.Seccond_layer.initiate(self.V2, self.b2, self.W01, self.W11)
        self.Third_layer.initiate(self.V3, self.b3, self.W12)

    def forward(self, test_data, target_data, clock):
        # 学習に使う配列の決定
        # print("test")
        memo_out = []
        #print("test_number:", test_number)
        #ix = test_data[test_number]

        out1 = None
        cont = 0
        dtT = int(1/self.dt)
        #print("dtT", dtT)
        #print("sequence_length", self.sequence_length)
        # for step in range(self.sequence_length):
        #input = ix[step].reshape([1, self.I])
        # if cont == 0:
        #clock = int(100 / self.dt)
        #print("clock", clock)
        out_memo = np.zeros((0, self.H))

        #est_number = np.random.randint(target_data.shape[0])

        for t in range(clock):
            if(t % 1000 == 0):
                print(test_data[0, t])
            input = test_data[0, t].reshape((1, self.I))
            v2, out1 = self.Seccond_layer.forward(input)
            v3, out2 = self.Third_layer.forward(out1)
            if t % dtT == 0:
                memo_out.append(out1)
                # print("memo")
        # plt.plot(self.b_lr_memo)

        self.out1 = out1
        self.out2 = out2

        return memo_out

    def output_H(self):
        return self.out1

    def output_O(self):
        return self.out2

    def setlr(self, lr, model=0):
        """
        model=0:SGD
        model=1:AdaGrad
        model=2:NormGrad
        """
        self.lr = lr
#       self.Seccond_layer.setlr(lr, model)
#        self.Third_layer.setlr(lr, model)

    def dx(self):
        return self.dx_memo

    def bo(self):
        return self.bo_memo

    def W_12(self):
        return self.W12_memo

    def set_swquwnce_length(self, sq):
        self.sequence_length = sq

    def sequence_prediction(self, x):
        # 学習に使う配列の決定
        out1 = self.h_prev[self.sequence_length - 1]
        input = np.array([[x[0][0]]])
        out1, container1 = self.Seccond_layer.forward(input, out1)
        self.h_prev[self.sequence_length - 1] = out1
        out2, container2 = self.Third_layer.forward(out1)
        return out2

    def set_NormGrad(self, NormGrad):
        # 勾配クリッピングのしきい値の変更
        self.Seccond_layer.change_NormGrad(NormGrad)

    def set_adjust(self, lr_index, min_loss):
        self.lr_index = lr_index
        self.min_loss = min_loss
