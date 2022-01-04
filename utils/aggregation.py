import copy

import torch


def fed_avg(w_global, w, count):
    """ FedAvg
        https://arxiv.org/pdf/1602.05629.pdf"""
    w_mul = []
    tmp = copy.deepcopy(w_global)

    for j in w:
        w_avg = copy.deepcopy(w[j])

        for i in w_avg.keys():
            if "user_embedding" in i or "embedding_user" in i or "user_bias" in i:
                tmp[i][j] = w_avg[i][j]
            else:
                w_avg[i] = torch.mul(w_avg[i], count[j])

        w_mul.append(w_avg)

    w_updated = copy.deepcopy(w_mul[0])

    for k in w_updated.keys():
        if "user_embedding" in k or "embedding_user" in k or "user_bias" in k:
            w_updated[k] = tmp[k]
        for i in range(1, len(w_mul)):
            w_updated[k] += w_mul[i][k]
        w_updated[k] = torch.div(w_updated[k], sum(count))

    return w_updated


def avg(w_global, w, count):
    w_mul = []
    tmp = copy.deepcopy(w_global)

    for j in w:
        w_avg = copy.deepcopy(w[j])

        for i in w_avg.keys():
            if "user_embedding" in i or "embedding_user" in i or "user_bias" in i:
                tmp[i][j] = w_avg[i][j]

        w_mul.append(w_avg)

    w_updated = copy.deepcopy(w_mul[0])

    for k in w_updated.keys():
        if "user_embedding" in k or "embedding_user" in k or "user_bias" in k:
            w_updated[k] = tmp[k]
        for i in range(1, len(w_mul)):
            w_updated[k] += w_mul[i][k]
        w_updated[k] = torch.div(w_updated[k], len(w))

    return w_updated
