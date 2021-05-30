from rnnclass import BPneuron, Loss, RNNneuron
import matplotlib.pyplot as plt
import numpy as np
from random import random
import matplotlib.pyplot as plt
from IPython.display import clear_output


class test_network:
    def __init__(self, H, O, W12_size):

        # 重みとバイアスの定義
        W12 = W12_size*np.random.rand(H, O)-W12_size/2
        b2 = np.zeros(O)

        # モデルの生成
        self.Third_layer = BPneuron(W12, b2)
        self.test_loss_layer = Loss()

        self.loss_memo = []
        self.b_lr_memo = []
        self.dx_memo = np.empty(0)
        self.bo_memo = np.empty(0)
        self.dbo_memo = np.empty(0)
        self.W12_memo = np.empty(0)

        self.lr = 0.01

        self.Who = None
        self.b = None

        # Abeloss用のlossの入れもの
        self.loss_container = 0

        # 学習率の減数率指数
        self.lr_index = 2

        # 誤差の最小範囲
        self.min_loss = 0.003

    def traning(self, test_data, target_data, epoch):
        # 学習に使う配列の決定
        # print("test")
        for i in range(epoch):
            test_number = np.random.randint(target_data.shape[0])
            ix = test_data[test_number]
            t = np.array([target_data[test_number]])
            out2, container2 = self.Third_layer.forward(ix)
            W, b, x, y, z = container2
            loss = self.test_loss_layer.forward(out2, t)
            self.loss_container += loss
            dout = self.test_loss_layer.backward()
            if i % 100 == 0 and i != 0:
                if (self.loss_container / 100/0.25) > self.min_loss:
                    New_lr = self.lr * \
                        (self.loss_container / 100/0.25) ** self.lr_index

            if i % 500 == 0 and i != 0:
                print("loss:", dout, "epoch:", i)
                print("_________")
                print("tnum", test_number)
                print("out2:", out2)
                print("t:", t)
                print("x:", ix)
                # print("lrs1:", lrs1)
                # print("lrs2:", lrs2)
                print("New_lr:", New_lr)
                print("_________")
                # self.b_lr_memo.append(lrs1[0][0][0])

            if i % 5000 == 0:
                clear_output()

            self.loss_memo.append(loss)
            dx, containe = self.Third_layer.backward(dout)

        self.Who = W
        self.b = b

        return self.loss_memo

    def output_w(self):
        return self.Who, self.b

    def setlr(self, lr, model=0):
        """
        model=0:SGD
        model=1:AdaGrad
        model=2:NormGrad
        """
        self.lr = lr
        self.Third_layer.setlr(lr, model)

    def dx(self):
        return self.dx_memo

    def bo(self):
        return self.bo_memo

    def W_12(self):
        return self.W12_memo

    def set_swquwnce_length(self, sq):
        self.sequence_length = sq

    def set_Wandb(self, W12, b2):
        self.Third_layer = BPneuron(W12, b2)
        self.test_loss_layer = Loss()
