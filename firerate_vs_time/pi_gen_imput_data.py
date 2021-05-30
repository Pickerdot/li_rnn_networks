import numpy as np
import test_rnn_realtime as net
import time

# 近似データの定義
high_value = 1.0
low_value = 0.0
category_num = 2
steps_num = 3


def make_input_4data(high_value, low_value, category_num):
    one_cate_data = np.empty((0, 0, category_num))
    high_element = np.array([[high_value]])
    low_element = np.array([[low_value]])
    much = np.array([1, 0])
    non_much = np.array([0, 1])

    input_data = np.empty((0, 4, category_num))
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
