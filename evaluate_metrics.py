import numpy as np
import pandas as pd
import torch
def hamming_loss(y_GT, predict):
    GT_size = np.sum(y_GT, axis=1)
    predict_label = np.zeros(predict.shape, dtype=int)
    sorted = predict.argsort()
    temp = 0

    for i in range(GT_size.shape[0]):
        index=sorted[i][-GT_size[i]:][::-1]
        predict_label[i][index]=1
        temp = temp + np.sum(y_GT[i] ^ predict_label[i])

    hmloss = temp/(y_GT.shape[0]*y_GT.shape[1])
    return hmloss

def FR(y_GT, predict):
    GT_size = np.sum(y_GT, axis=1)
    predict_label = np.zeros(predict.shape, dtype=int)
    sorted = predict.argsort()
    temp = 0

    for i in range(GT_size.shape[0]):
        index = sorted[i][-GT_size[i]:][::-1]
        predict_label[i][index] = 1
        if np.sum(y_GT[i] ^ predict_label[i])>0:
            temp = temp + 1

    fr = temp / y_GT.shape[0]
    return fr

# def TP_index(y_targets, predict):
#     GT_size = np.sum(y_targets, axis=1)
#     predict_label = np.zeros(predict.shape, dtype=int)
#     sorted = predict.argsort()
#     tp_list = []
#
#     for i in range(GT_size.shape[0]):
#
#         index = sorted[i][-GT_size[i]:][::-1]
#         predict_label[i][index] = 1
#         if np.sum(y_targets[i] ^ predict_label[i])==0:
#             tp_list.append(i)
#
#     return tp_list, predict_label

def TP_index(y_targets, predict):
    GT_size = np.sum(y_targets, axis=1)
    predict_label = np.zeros(predict.shape, dtype=int)
    sorted = predict.argsort()
    tp_flag = False

    for i in range(GT_size.shape[0]):
        index = sorted[i][-GT_size[i]:][::-1]
        predict_label[i][index] = 1
        if np.sum(y_targets[i] ^ predict_label[i])==0:
            tp_flag = True
    return tp_flag, predict_label


def nontargeted_TP_index(y_GT, predict, kvalue):
    predict_label = np.zeros(predict.shape, dtype=int)
    sorted = predict.argsort()
    flag = True

    for i in range(predict.shape[0]):
        index = sorted[i][-kvalue:][::-1]
        # index = torch.flip(index, [0])
        predict_label[i][index] = 1
        for j in index:
            if y_GT[i][j] == 1:
                flag = False
    return flag, predict_label

def UAP_TP_index(y_GT, predict, kvalue):
    predict_label = np.zeros(predict.shape, dtype=int)
    sorted = predict.argsort()
    index = sorted[:, -kvalue:][:, ::-1]
    b_new = np.where(y_GT == 1)
    data = pd.DataFrame({'id': list(b_new[0]), 'value': list(np.char.mod('%d', b_new[1]))})
    data_new = data.groupby(by='id').apply(lambda x: [' '.join(x['value'])])
    count = 0
    for i in range(y_GT.shape[0]):
        predict_label[i][index[i]] = 1
        set_a = set(index[i])
        set_b = set(list(map(int, data_new[i][0].split(' '))))
        if len(set_a.intersection(set_b)) == 0:
            count = count + 1
    fooling_rate = count / y_GT.shape[0]
    return fooling_rate, predict_label

def label_match(y_GT, predict,k_value):
    GT_size = np.sum(y_GT, axis=1)
    predict_label = np.zeros(predict.shape, dtype=int)
    sorted = predict.argsort()
    tp_list = []

    for i in range(GT_size.shape[0]):
        index = sorted[i][-k_value:][::-1]
        for j in index:
            if y_GT[j]==1:
                return False
        return True

def topk_acc_metric(y_GT, predict, kvalue):
    count = 0
    all_list = set(list(range(0, y_GT.shape[1])))
    for i in range(y_GT.shape[0]):
        GT_label = set(np.transpose(np.argwhere(y_GT[i]==1))[0])
        predict_index = predict[i].argsort()[-kvalue:][::-1]
        predict_complement_set = all_list - set(predict_index)
        count = count + int(predict_complement_set.issuperset(GT_label))
    return count / y_GT.shape[0]

def topk_acc_metric_1_to_10(y_GT, predict):
    count_1 = 0
    count_2 = 0
    count_3 = 0
    count_4 = 0
    count_5 = 0
    count_10 = 0
    all_list = set(list(range(0, y_GT.shape[1])))
    for i in range(y_GT.shape[0]):
        GT_label = set(np.transpose(np.argwhere(y_GT[i]==1))[0])
        predict_index_1 = predict[i].argsort()[-1:][::-1]
        predict_index_2 = predict[i].argsort()[-2:][::-1]
        predict_index_3 = predict[i].argsort()[-3:][::-1]
        predict_index_4 = predict[i].argsort()[-4:][::-1]
        predict_index_5 = predict[i].argsort()[-5:][::-1]
        predict_index_10 = predict[i].argsort()[-10:][::-1]
        predict_complement_set_1 = all_list - set(predict_index_1)
        predict_complement_set_2 = all_list - set(predict_index_2)
        predict_complement_set_3 = all_list - set(predict_index_3)
        predict_complement_set_4 = all_list - set(predict_index_4)
        predict_complement_set_5 = all_list - set(predict_index_5)
        predict_complement_set_10 = all_list - set(predict_index_10)
        count_1 = count_1 + int(predict_complement_set_1.issuperset(GT_label))
        count_2 = count_2 + int(predict_complement_set_2.issuperset(GT_label))
        count_3 = count_3 + int(predict_complement_set_3.issuperset(GT_label))
        count_4 = count_4 + int(predict_complement_set_4.issuperset(GT_label))
        count_5 = count_5 + int(predict_complement_set_5.issuperset(GT_label))
        count_10 = count_10 + int(predict_complement_set_10.issuperset(GT_label))
    return count_1 / y_GT.shape[0],count_2 / y_GT.shape[0],count_3 / y_GT.shape[0],count_4 / y_GT.shape[0],count_5 / y_GT.shape[0],count_10 / y_GT.shape[0]

