from rnnclass import BPneuron, Loss, RNNneuron
import matplotlib.pyplot as plt
import numpy as np
from random import random
import matplotlib.pyplot as plt
from IPython.display import clear_output


class test_network:
    def __init__(self, I, H, O, W01_size, W11_size, W12_size, sequence_length, w_path):
        # 出力ファイルの生成
        self.w_path = w_path
        if self.w_path is not None:
            text = "new data" + str(I) + "-" + str(H) + "-" + \
                str(O) + "|W01_size" + str(W01_size) + \
                "|W12_size" + str(W12_size)
            f = open(self.w_path, mode='w')
            f.write(text)
            f.close()

        # 重みとバイアスの定義
        W01 = W01_size*np.random.rand(I, H)-W01_size/2
        W11 = W11_size*np.random.rand(H, H)-W11_size/2
        W12 = W12_size*np.random.rand(H, O)-W12_size/2
        b1 = np.zeros(H)
        b2 = np.zeros(O)

        # モデルの生成
        self.Seccond_layer = RNNneuron(W01, W11, b1)
        self.Third_layer = BPneuron(W12, b2)
        self.test_loss_layer = Loss()
        self.I = I

        self.loss_memo = []
        self.b_lr_memo = []
        self.dx_memo = np.empty(0)
        self.bo_memo = np.empty(0)
        self.dbo_memo = np.empty(0)
        self.W12_memo = np.empty(0)

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

        self.Wih = None
        self.Whh = None
        self.Who = None
        self.bh = None
        self.bo = None

    def set_Wandb(self, Wih, Whh, Who, bh, bo):
        self.Seccond_layer = RNNneuron(Wih, Whh, bh)
        self.Third_layer = BPneuron(Who, bo)

    def traning(self, test_data, target_data, epoch):
        # 学習に使う配列の決定
        # print("test")
        if self.w_path is not None:
            f = open(self.w_path, mode='a')
        for i in range(epoch):
            test_number = np.random.randint(target_data.shape[0])
            ix = test_data[test_number]
            t = np.array([target_data[test_number]])
            out1 = None
            for step in range(self.sequence_length):
                input = ix[step].reshape([1, self.I])
                out1, container1 = self.Seccond_layer.forward(input, out1)
                # print(out1.shape)
                self.h_prev[step] = out1
            out2, container2 = self.Third_layer.forward(out1)
            if i % 100 == 0:
                Wi, Wh, bh, x, y, z = container1
                # lrs1 = self.Seccond_layer.viewlr()
                Wo, bo, x, y, z = container2
                # lrs2 = self.Third_layer.viewlr()
            loss = self.test_loss_layer.forward(out2, t)
            self.loss_container += loss
            dout = self.test_loss_layer.backward()
            if i % 100 == 0 and i != 0:
                if (self.loss_container / 100/0.25) > self.min_loss:
                    New_lr = self.lr * \
                        (self.loss_container / 100/0.25) ** self.lr_index

                else:
                    New_lr = 0.0001

                self.loss_container = 0
                self.Seccond_layer.change_lr(New_lr)
                self.Third_layer.change_lr(New_lr)

            if i % 500 == 0 and i != 0:
                print("loss:", dout, "epoch:", i)
                print("_________")
                #print("out1:", out1)
                print("out2:", out2)
                print("t:", t)
                print("x:", ix)
                # print("lrs1:", lrs1)
                # print("lrs2:", lrs2)
                print("New_lr:", New_lr)
                print("_________")
                # self.b_lr_memo.append(lrs1[0][0][0])
                clipping = self.Seccond_layer.clipper_Chech()
                if clipping == 1:
                    print("Clipping was done")

            if i % 5000 == 0:
                Wi, Wh, bh, x, y, z = container1
                Wo, bo, x, y, z = container2
                self.Wih = Wi
                self.Whh = Wh
                self.Who = Wo
                self.bh = bh
                self.bo = bo

            self.loss_memo.append(loss)
            dx, containe = self.Third_layer.backward(dout)
            for step in range(self.sequence_length):
                h_prev = np.array(
                    [self.h_prev[self.sequence_length - step - 1]])
                dinput, containe = self.Seccond_layer.backward(dx, h_prev)
            self.dx_memo = np.append(self.dx_memo, dx)
            self.bo_memo = np.append(self.bo_memo, bo)
            self.W12_memo = np.append(self.W12_memo, Wo)
            self.Seccond_layer.reset()
        # plt.plot(self.b_lr_memo)
        Wi, Wh, bh, x, y, z = container1
        Wo, bo, x, y, z = container2
        self.Wih = Wi
        self.Whh = Wh
        self.Who = Wo
        self.bh = bh
        self.bo = bo
        print("bo:", self.bo)

        return self.loss_memo

    def prediction(self, x, t, reset=True, w_path=None):
        self.w_path = w_path
        # 学習に使う配列の決定
        if self.w_path is not None:
            f = open(self.w_path, mode='a')
        out1 = None
        for step in range(self.sequence_length):
            input = np.array([[x[0][step]]])
            out1, container1 = self.Seccond_layer.forward(input, out1)
            self.h_prev[step] = out1
        out2, container2 = self.Third_layer.forward(out1)
        W, Wh, b, x, y, z = container1
        if self.w_path is not None:
            f = open(self.w_path, mode='a')
            w = "_________\prediction:"\
                + "t:"+str(t)\
                + "_________\nx: "+str(x)\
                + "\nW: "+str(W)\
                + "\nWh:"+str(Wh)\
                + "\nb:"+str(b)\
                + "\ny:"+str(y)\
                + "\nz:"+str(z)\
                + "_________\n"
            f.write(w)
        # lrs1 = self.Seccond_layer.viewlr()
        W, b, x, y, z = container2
        if self.w_path is not None:
            w = "_________\nx:"+str(x)+"\nW:"+str(
                W)+"\nb:"+str(b)+"\ny:"+str(y)+"\nz:"+str(z)+"_________\n"
            f.write(w)
        # lrs2 = self.Third_layer.viewlr()}
        loss = self.test_loss_layer.forward(out2, t)
        self.loss_container += loss
        dout = self.test_loss_layer.backward()
        if reset is True:
            self.Seccond_layer.reset()

        return out2, loss

    def setlr(self, lr, model=0):
        """
        model=0:SGD
        model=1:AdaGrad
        model=2:NormGrad
        """
        self.lr = lr
        self.Seccond_layer.setlr(lr, model)
        self.Third_layer.setlr(lr, model)

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

    def out_Wandb(self):
        return self.Wih, self.Whh, self.Who, self.bh, self.bo
