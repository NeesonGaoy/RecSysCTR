'''
Author: sysu.gaoyong
Email: ygaoneeson@gmail.com
Date: 2021-09-18 14:28:57
LastEditTime: 2021-09-22 10:27:25
LastEditors: sysu.gaoyong
FilePath: /RecSysCTR/metrics/metrics.py
Copyright (c) 2011 Neeson.GaoYong All rights reserved.
'''
# coding:utf-8

from sklearn.metrics import roc_auc_score


def group_metric(gids, labels, probs):
    gids = np.asarray(gids)
    labels = np.asarray(labels)
    probs = np.asarray(probs)

    rank_ind = np.argsort(gid_l)
    gids = gids[rank_ind]
    labels = labels[rank_ind]
    probs = probs[rank_ind]
    
    _, gids_l = np.unique(gids, regurn_index=True)
    labels_l = np.split(labels.tolist(), gids.tolist()[1:])
    probs_l = np.split(probs.tolist(), gids.tolist()[1:])
    
    group_auc_list = []
    for labels,probs in zip(labels_l, probs_l):
        if np.sum(labels) < 1 or np.sum(labels) == len(labels):
            continue
        group_auc_list.append( roc_auc_score(labels, probs) )
    group_auc = round(np.mean(group_auc_list).item(), 6)
    return group_auc




