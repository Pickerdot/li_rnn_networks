from rnnclass import BPneuron, Loss, RNNneuron
import numpy as np
from IPython.display import clear_output
from rnn_layer_unit import rnn_layer_unit
from non_layer_unit import non_layer_unit


class li_rnn_networks:
    def __init__(self, I, H, O, W01_size, W11_size, W12_size, sequence_length):
        # 重みとバイアスの定義
        self.W01 = W01_size*np.random.rand(I, H)-W01_size/2
        self.W11 = W11_size*np.random.rand(H, H)-W11_size/2
        self.W12 = W12_size*np.random.rand(H, O)-W12_size/2
        self.V2 = np.zeros((1, H))
        self.V3 = np.zeros((1, O))
        self.b1 = np.zeros(H)
        self.b2 = np.zeros(O)

        # モデルの生成
        self.Seccond_layer = RNNneuron(self.W01, self.W11, self.b1)
        self.Third_layer = BPneuron(self.W12, self.b2)
        self.test_loss_layer = Loss()
        self.I = I
        self.H = H
        self.O = O

        self.lSeccond_layer = rnn_layer_unit()
        self.lThird_layer = non_layer_unit()

        self.loss_memo = []
        self.loss_memo2 = []
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

    def def_parameter(self, Tau_H, Tau_O, dt, random_del, Step_length):
        self.random_del = random_del
        self.dt = dt
        self.Step_length = Step_length
        self.Tau_H = Tau_H
        self.Tau_O = Tau_O
        self.lSeccond_layer.def_parameter(Tau_H, self.dt, self.random_del)
        self.lThird_layer.def_parameter(Tau_O, self.dt, self.random_del)

        self.lSeccond_layer.initiate(self.V2, self.b1, self.W01, self.W11)
        self.lThird_layer.initiate(self.V3, self.b2, self.W12)

    def set_Wandb(self, W01, W11, W12, b1, b2):
        self.W01 = W01
        self.W11 = W11
        self.W12 = W12
        self.b1 = b1
        self.b2 = b2
        self.Seccond_layer = RNNneuron(self.W01, self.W11, self.b1)
        self.Third_layer = BPneuron(self.W12, self.b2)

    def output_W(self):
        return self.W01, self.W11, self.W12, self.b1, self.b2

    def network_reset(self):
        self.V2 = np.zeros((1, self.H))
        self.V3 = np.zeros((1, self.O))
        self.lSeccond_layer.initiate(self.V2, self.b1, self.W01, self.W11)
        self.lThird_layer.initiate(self.V3, self.b2, self.W12)

    def traning(self, test_data, target_data, epoch):
        self.test_data = test_data
        self.target_data = target_data
        # 学習に使う配列の決定
        # print("test")
        for i in range(epoch):
            test_number = np.random.randint(target_data.shape[0])
            ix = test_data[test_number]
            t = np.array([target_data[test_number]])
            out1 = None
            for step in range(self.sequence_length):
                input_data = ix[step].reshape([1, self.I])
                out1, container1 = self.Seccond_layer.forward(input_data, out1)
                # print(out1.shape)
                self.h_prev[step] = out1
            out2, container2 = self.Third_layer.forward(out1)

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
                print("loss:", np.linalg.norm(out2-t),  "epoch:", i)
                print("_________")
                print("out1:", out1)
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
                clear_output()

            self.loss_memo.append((np.linalg.norm(out2-t)**2)/2)
            dx, containe = self.Third_layer.backward(dout)
            for step in range(self.sequence_length):
                h_prev = np.array(
                    [self.h_prev[self.sequence_length - step - 1]])
                dinput, containe = self.Seccond_layer.backward(dx, h_prev)
            # self.dx_memo = np.append(self.dx_memo, dx)
            # self.bo_memo = np.append(self.bo_memo, b)
            # self.W12_memo = np.append(self.W12_memo, W)
            self.Seccond_layer.reset()
        # plt.plot(self.b_lr_memo)

        self.W01, self.W11, self.b1, x, y, z = container1
        self.W12, self.b2, x, y, z = container2

        return self.loss_memo

    def to_li_para(self, epoch, interruption=True):
        self.loss_memo2 = []
        self.gene_li_out()
        # 学習に使う配列の決定
        # print("test")
        for i in range(epoch):
            test_number = np.random.randint(self.itarget_data.shape[0])
            ix = self.itest_data[test_number]
            t = np.array([self.itarget_data[test_number]])
            out2, container2 = self.Third_layer.forward(ix)
            W, b, x, y, z = container2
            loss = self.test_loss_layer.forward(out2, t)
            if (self.loss_container < loss):
                self.loss_container = loss
            dout = self.test_loss_layer.backward()
            if i % 100 == 0 and i != 0:
                if (interruption is True):
                    if ((self.loss_container) < 0.2):
                        print(self.loss_container)
                        return self.loss_memo2
                if (self.loss_container / 0.25) > self.min_loss:
                    New_lr = self.lr * \
                        (self.loss_container / 0.25) ** self.lr_index

            if i % 500 == 0 and i != 0:
                print("loss:", dout, "epoch:", i)
                print("_________")
                print("tnum", test_number)
                print("out2:", out2)
                print("t:", t)
                print("x:", ix)
                print("New_lr:", New_lr)
                print("_________")

            if i % 5000 == 0:
                clear_output()

            self.loss_memo2.append(loss)
            dx, containe = self.Third_layer.backward(dout)

        self.Who = W
        self.b2 = b

        self.Seccond_layer = RNNneuron(self.W01, self.W11, self.b1)
        self.Third_layer = BPneuron(self.W12, self.b2)

        return self.loss_memo2

    def li_forward(self, test_data, target_data, test_num=None):
        memo_out1 = np.empty((0, self.H))
        memo_out2 = np.empty((0, self.O))
        self.network_reset()

        out1 = None
        cont = 0
        dtT = int(1/self.dt)

        if (test_num != None):
            test_data = test_data[test_num]
            print("test_num:", test_num)

        for seaquence in range(self.sequence_length):
            input_data = test_data[seaquence].reshape((1, self.I))
            self.Step_length = 100 # 用削除
            if seaquence == 1:
                self.Step_length = 200
            for t in range(self.Step_length * dtT):
                ix = input_data.reshape((1, self.I))
                v2, out1 = self.lSeccond_layer.forward(ix)
                v3, out2 = self.lThird_layer.forward(out1)
                if t % dtT == 0:
                    memo_out1 = np.vstack((memo_out1, out1))
                    memo_out2 = np.vstack((memo_out2, out2))
                # print("memo")
        # plt.plot(self.b_lr_memo)

        self.out1 = out1
        self.out2 = out2

        return memo_out1, memo_out2

    def li_forward_realtime(self, test_data, target_data, test_num=None):
        memo_out1 = np.empty((0, self.H))
        memo_out2 = np.empty((0, self.O))
        self.network_reset()

        out1 = None
        cont = 0
        dtT = int(1/self.dt)

        for line in test_data[test_num]:
            ix = line.reshape((1, self.I))
            v2, out1 = self.lSeccond_layer.forward(ix)
            v3, out2 = self.lThird_layer.forward(out1)
            if cont % dtT == 0:
                memo_out1 = np.vstack((memo_out1, out1))
                memo_out2 = np.vstack((memo_out2, out2))
            cont += 1

        self.out1 = out1
        self.out2 = out2

        return memo_out1, memo_out2

    def forward(self, test_data, target_data, test_number):
        # 学習に使う配列の決定
        # print("test")
        memo_out = []
        print("test_number:", test_number)
        ix = test_data[test_number]
        print("ix:", ix)
        t = np.array([target_data[test_number]])
        print("targer:", t)
        out1 = None
        cont = 0
        dtT = int(1/self.dt)
        #print("dtT", dtT)
        for step in range(self.sequence_length):
            input = ix[step].reshape([1, self.I])
            if cont == 0:
                clock = int(100 / self.dt)
                print("clock", clock)
                for t in range(clock):
                    v2, out1 = self.lSeccond_layer.forward(input)
                    v3, out2 = self.lThird_layer.forward(out1)
                    if t % dtT == 0:
                        memo_out.append(out2)
                        # print("memo")
            else:
                clock = int(self.Step_length / self.dt)
                print("clock", clock)
                for t in range(clock):
                    v2, out1 = self.lSeccond_layer.forward(input)
                    v3, out2 = self.lThird_layer.forward(out1)
                    if t % dtT == 0:
                        memo_out.append(out2)
                        # print("memo2")
            # print(out1.shape)
            self.h_prev[step] = out1

            cont += 1
        # self.def_parameter(self.Tau, self.dt,
        #                    self.random_del, self.Step_length)
        # plt.plot(self.b_lr_memo)

        self.out1 = out1

        return memo_out

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
        input_data = np.array([[x[0][0]]])
        out1, container1 = self.Seccond_layer.forward(input_data, out1)
        self.h_prev[self.sequence_length - 1] = out1
        out2, container2 = self.Third_layer.forward(out1)
        return out2

    def set_NormGrad(self, NormGrad):
        # 勾配クリッピングのしきい値の変更
        self.Seccond_layer.change_NormGrad(NormGrad)

    def set_adjust(self, lr_index, min_loss):
        self.lr_index = lr_index
        self.min_loss = min_loss

    def gene_li_out(self):
        self.itest_data = np.empty((0, 1, self.H))
        self.itarget_data = np.empty((0, self.I))
        for i in range(self.test_data.shape[0]):
            self.def_parameter(self.Tau_H, self.Tau_O, self.dt,
                               self.random_del, self.Step_length)
            self.network_reset()
            memo = self.li_forward(self.test_data, self.target_data, i)
            test = np.round(self.output_H(), decimals=3)
            test = np.array([test])
            self.itest_data = np.vstack((self.itest_data, test))
            addtarget = self.target_data[i]
            print(addtarget)
            self.itarget_data = np.vstack((self.itarget_data, addtarget))

        print(self.itest_data.shape)
        print(self.itarget_data.shape)

    def output_H(self):
        return self.out1

    def output_O(self):
        return self.out2

    def out_Wandb(self):
        return self.W01, self.W11, self.W12, self.b1, self.b2

    def to_li_para_realtime(self, epoch, rinput_data, rtarget_data, interruption=True):
        self.loss_memo2 = []
        self.gene_li_out_realtime(rinput_data, rtarget_data)
        # 学習に使う配列の決定
        # print("test")
        for i in range(epoch):
            out1, out2 = self.gene_li_out_realtime(rinput_data, rtarget_data)
            for i in range(out2.shape[0]):
                print(out1[0])



            if i % 500 == 0 and i != 0:
                print("loss:", dout, "epoch:", i)
                print("_________")
                print("tnum", test_number)
                print("out2:", out2)
                print("t:", t)
                print("x:", ix)
                print("New_lr:", New_lr)
                print("_________")

            if i % 5000 == 0:
                clear_output()

            self.loss_memo2.append(loss)
            dx, containe = self.Third_layer.backward(dout)

        self.Who = W
        self.b2 = b

        self.Seccond_layer = RNNneuron(self.W01, self.W11, self.b1)
        self.Third_layer = BPneuron(self.W12, self.b2)

        return self.loss_memo2

    def gene_li_out_realtime(self, rinput_data, rtarget_data):
        self.itest_data = np.empty((0, 1, self.H))
        self.itarget_data = np.empty((0, self.I))
        for i in range(self.test_data.shape[0]):
            self.def_parameter(self.Tau_H, self.Tau_O, self.dt,
                               self.random_del, self.Step_length)
            self.network_reset()
            outH, outO = self.li_forward_realtime(rinput_data, rtarget_data, i)
            test = np.round(self.output_H(), decimals=3)
            test = np.array([test])
            self.itest_data = np.vstack((self.itest_data, test))
            addtarget = self.target_data[i]
            print(addtarget)
            self.itarget_data = np.vstack((self.itarget_data, addtarget))


def task_3back(high_value, low_value, category_num):
    one_cate_data = np.empty((0, 0, category_num))
    high_element = np.array([[high_value]])
    low_element = np.array([[low_value]])
    t = np.zeros((category_num, category_num))
    for i in range(category_num):
        t[i, i] = 1

    input_data = np.empty((0, 3, category_num))
    target_data = np.empty((0, category_num))
    for q in range(category_num):
        for i in range(category_num):
            for j in range(category_num):
                one_cate_data = np.empty((0, category_num))
                one_low_data = np.empty((1, 0))
                for l in range(category_num):
                    if l == q:
                        one_low_data = np.hstack(
                            (one_low_data, high_element))
                    else:
                        one_low_data = np.hstack(
                            (one_low_data, low_element))
                one_cate_data = np.vstack((one_cate_data, one_low_data))
                one_low_data = np.empty((1, 0))
                for l in range(category_num):
                    if l == i:
                        one_low_data = np.hstack(
                            (one_low_data, high_element))
                    else:
                        one_low_data = np.hstack(
                            (one_low_data, low_element))
                one_cate_data = np.vstack((one_cate_data, one_low_data))
                one_low_data = np.empty((1, 0))
                for l in range(category_num):
                    if l == j:
                        one_low_data = np.hstack(
                            (one_low_data, high_element))
                    else:
                        one_low_data = np.hstack(
                            (one_low_data, low_element))
                one_cate_data = np.vstack((one_cate_data, one_low_data))
                one_low_data = np.empty((1, 0))

                one_cate_data_3 = np.array([one_cate_data])
                input_data = np.vstack((input_data, one_cate_data_3))
                addtarget = t[q]
                target_data = np.vstack((target_data, addtarget))

    return input_data, target_data


def f3_23back(high_value, low_value, category_num, steps_num):
    one_cate_data = np.empty((0, 0, category_num))
    high_element = np.array([[high_value]])
    low_element = np.array([[low_value]])
    t = np.zeros((category_num, category_num))
    for i in range(category_num):
        t[i, i] = 1

    input_data = np.empty((0, steps_num, category_num))
    target_data = np.empty((0, category_num*2))
    for q in range(category_num):
        for i in range(category_num):
            for j in range(category_num):
                one_cate_data = np.empty((0, category_num))
                one_low_data = np.empty((1, 0))
                for l in range(category_num):
                    if l == q:
                        one_low_data = np.hstack(
                            (one_low_data, high_element))
                    else:
                        one_low_data = np.hstack(
                            (one_low_data, low_element))
                one_cate_data = np.vstack((one_cate_data, one_low_data))
                one_low_data = np.empty((1, 0))
                for l in range(category_num):
                    if l == i:
                        one_low_data = np.hstack(
                            (one_low_data, high_element))
                    else:
                        one_low_data = np.hstack(
                            (one_low_data, low_element))
                one_cate_data = np.vstack((one_cate_data, one_low_data))
                one_low_data = np.empty((1, 0))
                for l in range(category_num):
                    if l == j:
                        one_low_data = np.hstack(
                            (one_low_data, high_element))
                    else:
                        one_low_data = np.hstack(
                            (one_low_data, low_element))
                one_cate_data = np.vstack((one_cate_data, one_low_data))
                one_low_data = np.empty((1, 0))

                one_cate_data_3 = np.array([one_cate_data])
                input_data = np.vstack((input_data, one_cate_data_3))
                addtarget = np.hstack((t[q], t[i]))
                target_data = np.vstack((target_data, addtarget))

    return input_data, target_data

