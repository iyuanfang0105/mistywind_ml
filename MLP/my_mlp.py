#coding=utf-8
import numpy as np

with open('/Users/wind/WORK/public_data_set/uip_adv/labelled.ds', 'r') as inf:
    data_set = []
    for l in inf.readlines():
        tmp = l.strip().split('##yf##')
        assert len(tmp) == 2, 'ERROR line format'
        data_set.append((tmp[0].split(" "), tmp[1]))

features_list = []
for item in data_set:
    for feat in item[0]:
        if feat not in features_list:
            features_list.append(feat)

features_map = {}
for idx, feat in enumerate(features_list):
    features_map[feat] = idx

train_ds = []
for l in data_set:
    feat = np.zeros((1, len(features_map)), dtype=np.float32)
    for i in l[0]:
        feat[0, features_map[i]] = 1
    train_ds.append(feat)

train_ds = np.asarray(train_ds)

print(train_ds.size)