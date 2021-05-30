import numpy as np
from random import random
import matplotlib.pyplot as plt
from IPython.display import clear_output
from rnn_layer_unit import rnn_layer_unit
from non_layer_unit import non_layer_unit
from rnnclass import Loss


class HO_layer:
    def __init__(self, H, O, Who_size):
        self.H = H
        self.O = O

        self.Who = Who_size * np.random.rand(H, O) - Who_size / 2
        self.Vo = np.zeros((1, O))
        self.So = np.zeros((1, O))
        self.bo = np.zeros((1, O))

        # 層の定義
        self.HO_layer = non_layer_unit()
        self.HO_layer.initiate(self.Vo, self.bo, self.Who)

        # Li用パラメータ
        self.random_del = 1
        self.dt = 0.01
        self.dtT = int(1/self.dt)
        self.Tau = 100
        self.Step_length = 100
        self.HO_layer.def_parameter(self.Tau, self.dt, self.random_del)

        # 誤差記録用のlist
        self.memo_loss = []

        self.memo_out = []

        self.lr = 0.01

        # Abeloss用のlossの入れもの
        self.loss_container = 0

        # 学習率の減数率指数
        self.lr_index = 2

        # 誤差の最小範囲
        self.min_loss = 0.003

    def def_parameter(self, Tau, dt, random_del, Step_Length):
        self.random_del = random_del
        self.dt = dt
        self.dtT = int(1/self.dt)
        self.Tau = Tau
        self.Step_length = Step_Length
        self.HO_layer.def_parameter(self.Tau, self.dt, self.random_del)

    def set_Wandb(self, Who):
        self.Who = Who
        self.HO_layer.initiate(self.Vo, self.bo, self.Who)

    def network_reset(self):
        self.Vo = np.zeros((1, self.O))
        self.HO_layer.initiate(self.Vo, self.bo, self.Who)

    def forward(self, input):
        clock = int(100 / self.dt)
        for t in range(clock):
            self.Vo, self.So = self.HO_layer.forward(input)

    def forward_dry(self, input):
        self.Vo, self.So = self.HO_layer.forward_dry(input)

    def traning(self, test_data, target_data, epoch):
        data_num = target_data.shape[0]
        for i in range(epoch):
            test_number = np.random.randint(data_num)
            input = test_data[test_number]
            target = np.array([target_data[test_number]])
            self.forward(input)
            self.HO_layer.loss(target)

            if i % 1000 == 0 and i != 0:
                print("loss:", self.HO_layer.output_loss, "epoch:", i)
                print("_________")

            if i % 5000 == 0:
                clear_output()

            self.HO_layer.backward(target)
            self.memo_loss.append(self.HO_layer.output_loss())
            self.network_reset()
        return self.memo_loss

    def traning_dry(self, test_data, target_data, epoch):
        data_num = target_data.shape[0]
        for i in range(epoch):
            test_number = np.random.randint(data_num)
            input = test_data[test_number]
            target = np.array([target_data[test_number]])
            self.forward_dry(input)
            self.HO_layer.loss(target)

            if i % 100 == 0 and i != 0:
                print("loss:", self.HO_layer.output_loss(target), "epoch:", i)
                print("_________")

            if i % 5000 == 0:
                clear_output()

            self.HO_layer.backward(target)
            self.memo_loss.append(self.HO_layer.output_loss(target))
            self.network_reset()
        return self.memo_loss

    def predict(self, input):
        clock = int(100 / self.dt)
        for t in range(clock):
            self.Vo, self.So = self.HO_layer.forward(input)
            if t % self.dtT == 0:
                self.memo_out.append(self.So)

    def predict_dry(self, input):
        self.Vo, self.So = self.HO_layer.forward_dry(input)

    def predict_result(self):
        return self.So, self.memo_out
