import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import test_rnn_realtime as net
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import test_rnn_realtime as net
import traning as tr
import time

# 近似データの定義
high_value = 1.0
low_value = 0.0
category_num = 2
steps_num = 3


def make_input_4data(high_value, low_value, category_num, steps_num):
    one_cate_data = np.empty((0, 0, category_num))
    high_element = np.array([[high_value]])
    low_element = np.array([[low_value]])
    much = np.array([1, 0])
    non_much = np.array([0, 1])

    input_data = np.empty((0, steps_num, category_num))
    target_data = np.empty((0, 2))
    for q in range(category_num):
        for i in range(category_num):
            for j in range(category_num):
                for k in range(category_num):
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
                    for l in range(category_num):
                        if l == k:
                            one_low_data = np.hstack(
                                (one_low_data, high_element))
                        else:
                            one_low_data = np.hstack(
                                (one_low_data, low_element))
                    one_cate_data = np.vstack((one_cate_data, one_low_data))

                    one_cate_data_3 = np.array([one_cate_data])
                    input_data = np.vstack((input_data, one_cate_data_3))
                    if i == j and i == k and q == j:
                        addtarget = much
                    else:
                        addtarget = non_much
                    target_data = np.vstack((target_data, addtarget))

    return input_data, target_data


def make_input_3data(high_value, low_value, category_num, steps_num):
    one_cate_data = np.empty((0, 0, category_num))
    high_element = np.array([[high_value]])
    low_element = np.array([[low_value]])
    much = np.array([1, 0])
    non_much = np.array([0, 1])

    input_data = np.empty((0, steps_num, category_num))
    target_data = np.empty((0, 2))
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
                if i == j and q == j:
                    addtarget = much
                else:
                    addtarget = non_much
                target_data = np.vstack((target_data, addtarget))

    return input_data, target_data


def make_input_2data(high_value, low_value, category_num, steps_num):
    one_cate_data = np.empty((0, 0, category_num))
    high_element = np.array([[high_value]])
    low_element = np.array([[low_value]])
    much = np.array([1, 0])
    non_much = np.array([0, 1])

    input_data = np.empty((0, steps_num, category_num))
    target_data = np.empty((0, 2))
    for q in range(category_num):
        for i in range(category_num):

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

            one_cate_data_3 = np.array([one_cate_data])
            input_data = np.vstack((input_data, one_cate_data_3))
            if i == q:
                addtarget = much
            else:
                addtarget = non_much
            target_data = np.vstack((target_data, addtarget))

    return input_data, target_data


def add_traning_data(input_data, target_data, category_num, chunk):
    much = np.array([1, 0])
    non_much = np.array([0, 1])
    for r in range(category_num ** (chunk - 1) - 2):
        for i in range(category_num):
            for j in range(1, category_num):
                non_much_num = ((i + j) % category_num) * category_num ** 3 + \
                    i * category_num ** 2 + i * category_num + i
                print(input_data[non_much_num])
                input_data = np.vstack(
                    (input_data, np.array([input_data[non_much_num]])))
                target_data = np.vstack(
                    (target_data, np.array([target_data[non_much_num]])))

                test_num = i * category_num ** 3 + i * category_num ** 2 + i * category_num + i
                print(input_data[test_num])
                for ex in range(1):
                    input_data = np.vstack(
                        (input_data, np.array([input_data[test_num]])))
                    target_data = np.vstack(
                        (target_data, np.array([target_data[test_num]])))
    return input_data, target_data


def iadd_traning_data(input_data, target_data, category_num, steps_num):
    much = np.array([1, 0])
    non_much = np.array([0, 1])
    iinput_data = np.empty((0, steps_num, category_num))
    itarget_data = np.empty((0, 2))
    for r in range(1):
        for i in range(category_num):
            for j in range(1, category_num):
                non_much_num = ((i + j) % category_num) * category_num ** 3 + \
                    i * category_num ** 2 + i * category_num + i
                print("non_much_num:", non_much_num)
                print(input_data[non_much_num])
                for ex in range(1):
                    iinput_data = np.vstack(
                        (iinput_data, np.array([input_data[non_much_num]])))
                    itarget_data = np.vstack(
                        (itarget_data, np.array([target_data[non_much_num]])))

                test_num = i * category_num ** 3 + i * category_num ** 2 + i * category_num + i
                print(input_data[test_num])
                for ex in range(1):
                    iinput_data = np.vstack(
                        (iinput_data, np.array([input_data[test_num]])))
                    itarget_data = np.vstack(
                        (itarget_data, np.array([target_data[test_num]])))
    return iinput_data, itarget_data


def iadd_traning_3data(input_data, target_data, category_num, steps_num):
    much = np.array([1, 0])
    non_much = np.array([0, 1])
    iinput_data = np.empty((0, steps_num, category_num))
    itarget_data = np.empty((0, 2))
    for r in range(1):
        for i in range(category_num):
            for j in range(1, category_num):
                non_much_num = ((i + j) % category_num) * category_num ** 2 + \
                    i * category_num + i
                print("non_much_num:", non_much_num)
                print(input_data[non_much_num])
                iinput_data = np.vstack(
                    (iinput_data, np.array([input_data[non_much_num]])))
                itarget_data = np.vstack(
                    (itarget_data, np.array([target_data[non_much_num]])))

                test_num = i * category_num ** 2 + i * category_num + i
                print(input_data[test_num])
                for ex in range(1):
                    iinput_data = np.vstack(
                        (iinput_data, np.array([input_data[test_num]])))
                    itarget_data = np.vstack(
                        (itarget_data, np.array([target_data[test_num]])))
    return iinput_data, itarget_data


def iadd_traning_2data(input_data, target_data, category_num, steps_num):
    much = np.array([1, 0])
    non_much = np.array([0, 1])
    iinput_data = np.empty((0, steps_num, category_num))
    itarget_data = np.empty((0, 2))
    for r in range(1):
        for i in range(category_num):
            for j in range(1, category_num):
                non_much_num = ((i + j) % category_num) * category_num ** 1 + i
                print("non_much_num:", non_much_num)
                print(input_data[non_much_num])
                iinput_data = np.vstack(
                    (iinput_data, np.array([input_data[non_much_num]])))
                itarget_data = np.vstack(
                    (itarget_data, np.array([target_data[non_much_num]])))

                test_num = i * category_num ** 1 + i
                print(input_data[test_num])
                for ex in range(1):
                    iinput_data = np.vstack(
                        (iinput_data, np.array([input_data[test_num]])))
                    itarget_data = np.vstack(
                        (itarget_data, np.array([target_data[test_num]])))
    return iinput_data, itarget_data


def make_test_data(input_data, target_data, Tau, Wih_size, Whh_size, Who_size, I, H, O, category_num, flug_test=False):
    Wih = Wih_size * np.random.rand(I, H) - Wih_size/2
    Whh = Whh_size * np.random.rand(H, H) - Whh_size/2
    Who = Who_size * np.random.rand(H, O) - Who_size/2
    bh = np.zeros((1, H))
    bo = np.zeros((1, O))
    test_net = net.test_network(I, H, O, 1, 1, 1, 3)
    test_net.set_Wandb(Wih, Whh, Who, bh, bo)
    step_length = 100
    test_net.def_parameter(Tau, 0.01, 1, step_length)
    test_net.network_reset()
    memo = test_net.forward(input_data, target_data, 0)
    test0 = np.round(test_net.output_H(), decimals=3)
    print_memo(memo)
    plt.figure()
    step_length = 100
    test_net.def_parameter(Tau, 0.01, 1, step_length)
    test_net.network_reset()
    memo = test_net.forward(input_data, target_data, 2)
    test2 = np.round(test_net.output_H(), decimals=3)
    print_memo(memo)
    plt.figure()
    step_length = 100
    test_net.def_parameter(Tau, 0.01, 1, step_length)
    test_net.network_reset()
    memo = test_net.forward(input_data, target_data, 4)
    test1 = np.round(test_net.output_H(), decimals=3)
    test1
    print_memo(memo)
    plt.figure()
    step_length = 100
    test_net.def_parameter(Tau, 0.01, 1, step_length)
    test_net.network_reset()
    memo = test_net.forward(input_data, target_data, 8)
    test3 = np.round(test_net.output_H(), decimals=3)
    print_memo(memo)
    plt.figure()
    ss = 10
    sr = int(H/ss)
    print("********")
    ptest0 = test0.reshape(ss, sr)
    sns.heatmap(ptest0)
    plt.figure()
    print("********")
    ptest1 = test1.reshape(ss, sr)
    sns.heatmap(ptest1)
    plt.figure()
    print("********")
    ptest2 = test2.reshape(ss, sr)
    sns.heatmap(ptest2)
    plt.figure()
    print("********")
    ptest3 = test3.reshape(ss, sr)
    sns.heatmap(ptest3)
    plt.figure()
    print("********")
    del_03 = abs(ptest0-ptest1)
    sns.heatmap(del_03, vmax=0.4)
    print("********")
    plt.figure()
    del_03 = abs(ptest0-ptest2)
    sns.heatmap(del_03, vmax=0.4)
    print("********")
    plt.figure()
    del_03 = abs(ptest0-ptest3)
    sns.heatmap(del_03, vmax=0.4)
    print("********")
    plt.figure()
    test_data = np.empty((0, 1, H))
    itarget_data = np.empty((0, 2))
    if flug_test == True:
        return
    for i in range(category_num):
        for j in range(category_num):
            for k in range(category_num):
                step_length = 100
                test_net.def_parameter(Tau, 0.01, 1, step_length)
                test_net.network_reset()
                test_num = i * category_num * category_num + j * category_num + k
                memo = test_net.forward(input_data, target_data, test_num)
                test = np.round(test_net.output_H(), decimals=3)
                test = np.array([test])
                test_data = np.vstack((test_data, test))
                addtarget = target_data[test_num]
                itarget_data = np.vstack((itarget_data, addtarget))
    for r in range(ccategory_num ** 3 +
                   i * ategory_num * (category_num)):
        for i in range(category_num):
            j, k = i, i
            test_num = i * category_num * category_num + j * category_num + k
            step_length = 100
            test_net.def_parameter(Tau, 0.01, 1, step_length)
            test_net.network_reset()
            memo = test_net.forward(input_data, target_data,  test_num)
            test = np.round(test_net.output_H(), decimals=3)
            test = np.array([test])
            test_data = np.vstack((test_data, test))
            addtarget = target_data[test_num]
            itarget_data = np.vstack((itarget_data, addtarget))
    return test_data, itarget_data


def traning(test_data, itarget_data, H, O, del_W_h, chunk):
    test_network = tr.test_network(H, O, 1)
    test_network.setlr(0.05)
    itarget_data.shape
    memo = test_network.traning(test_data, itarget_data, 300000)
    plt.plot(memo)
    name = "data/del_W_h:" + str(del_W_h) + \
        " chunk:" + str(chunk) + " H:" + str(H)
    plt.savefig(name)
    plt.figure()


def f3_2back(high_value, low_value, category_num, steps_num):
    one_cate_data = np.empty((0, 0, category_num))
    high_element = np.array([[high_value]])
    low_element = np.array([[low_value]])
    t = np.zeros((category_num, category_num))
    for i in range(category_num):
        t[i, i] = 1

    input_data = np.empty((0, steps_num, category_num))
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
                addtarget = t[i]
                target_data = np.vstack((target_data, addtarget))

    return input_data, target_data


def f3_3back(high_value, low_value, category_num, steps_num):
    one_cate_data = np.empty((0, 0, category_num))
    high_element = np.array([[high_value]])
    low_element = np.array([[low_value]])
    t = np.zeros((category_num, category_num))
    for i in range(category_num):
        t[i, i] = 1

    input_data = np.empty((0, steps_num, category_num))
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
