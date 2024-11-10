# conda activate /home/zhengweiguo/miniconda3/envs/lcj_bert
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from binary_io import *
import json
import os
import argparse

argparse = argparse.ArgumentParser()

dataset = 'imagenet'
dataset = 'gist1m'
dataset = 'wikipedia'
dataset = 'deep100m'
dataset = 'datacomp-image.base'
config = json.loads(open('config.json').read())
M, efs, threshold = config[dataset]["M"], config[dataset]["efs"], config[dataset]["threshold"]

dim = config[dataset]["dim"]
efc = 1000
ck_ts = 1000
k = 1000
query_only = True

data_prefix = '/data/disk1/liuchengjun/HNNS/sample/'
checkpoint_prefix = '/data/disk1/liuchengjun/HNNS/checkpoint/'
prefix = f'{dataset}.M_{M}.efc_{efc}.efs_{efs}.ck_ts_{ck_ts}.ncheck_100.recall@{k}'
checkpoint_regression_path = f'{checkpoint_prefix}{prefix}.thr_{threshold}.regression.{"qonly." if query_only else ""}txt'

train_feature = fvecs_read(f'{data_prefix}{prefix}.train_feats_nn.fvecs')
test_feature = fvecs_read(f'{data_prefix}{prefix}.test_feats_nn.fvecs')[:, :]
if query_only:
    train_feature = train_feature[:, :dim]
    test_feature = test_feature[:, :dim]

train_number_recall = ivecs_read(f'{data_prefix}{prefix}.train_label.ivecs')[:, 0]
train_comps = ivecs_read(f'{data_prefix}{prefix}.train_label.ivecs')[:, 1]
train_label_regression = np.log2(train_comps + 1)
test_number_recall = ivecs_read(f'{data_prefix}{prefix}.test_label.ivecs')[:, 0]
test_comps = ivecs_read(f'{data_prefix}{prefix}.test_label.ivecs')[:, 1]
test_label_regression = np.log2(test_comps + 1)

# print(len(train_feature), len(train_label_regression))
print(np.mean(test_label_regression), np.mean(train_label_regression))
# print(len(train_feature[0]), train_feature[0][-101:])

df = pd.DataFrame()

offset = 0
feat_query = train_feature[:, offset: offset + dim]
query_cols = [f"query_{i}" for i in range(dim)]
df_query = pd.DataFrame(feat_query, columns=query_cols)
df = pd.concat([df, df_query], axis = 1)
offset += dim

if offset < len(train_feature[0]):
    feat_dist = train_feature[:, offset: offset + 100]
    dist_cols = [f"dist_{i}" for i in range(100)]
    df_dist = pd.DataFrame(feat_dist, columns=dist_cols)
    df = pd.concat([df, df_dist], axis = 1)
    offset += 100
if offset < len(train_feature[0]):
    feat_update = train_feature[:, offset: offset + 3]
    update_cols = [f"update_{i}" for i in range(3)]
    df_update = pd.DataFrame(feat_update, columns=update_cols)
    df = pd.concat([df, df_update], axis = 1)
    offset += 3

assert offset == len(train_feature[0])

# df = pd.concat([df_query, df_dist, df_update], axis = 1)

##################################################  ##################################################

params_regression = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mse',
    'learning_rate': 0.05,
    'num_boost_round': 3000,
    'verbose': 1,
    "num_threads": 96
}

##################################################  ##################################################

if os.path.exists(checkpoint_regression_path):
    print(f'[Checkpoint] {checkpoint_regression_path} exist!')
    gbm_regression = lgb.Booster(model_file = checkpoint_regression_path)
    print('[Checkpoint] Loaded!')
else:
    print(f'[Checkpoint] {checkpoint_regression_path} not exist!')
    print('[Checkpoint] Training!')
    gbm_regression = lgb.train(params_regression, lgb.Dataset(df.values, label=train_label_regression))
    gbm_regression.save_model(checkpoint_regression_path)
    print('[Checkpoint] Done!')

train_pred = gbm_regression.predict(train_feature)
sorted_train_pred = np.sort(train_pred)

importance = gbm_regression.feature_importance()
feature_names = df.columns
query_imp, update_imp, dist_imp = 0, 0, 0
for f, imp in zip(feature_names, importance):
    if f.startswith('query'):
        query_imp += imp
    elif f.startswith('update'):
        update_imp += imp
    elif f.startswith('dist'):
        dist_imp += imp
    # print(f, imp)
print(f'importance query: {query_imp / dim}, update: {update_imp / 3}, dist: {dist_imp / 100}')

##################################################  ##################################################

from sklearn.metrics import recall_score
import time
import numpy as np
from sklearn.metrics import recall_score, roc_auc_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.integrate import trapezoid as trapz

start = time.time()
label_pred = gbm_regression.predict(test_feature)
end = time.time()

for i in range(100):
    print(f'{i}: {label_pred[i]} | {test_label_regression[i]}')
# mse loss
from sklearn.metrics import mean_squared_error
print(f'mse: {mean_squared_error(test_label_regression, label_pred):.2f}')

##################################################  ##################################################

pct50 = np.percentile(label_pred, 50)
label_pred_sorted = np.sort(label_pred)
step = 0.2
scores = 0.0
for p in np.arange(0, 100 + step, step):
    pct_thr = label_pred_sorted[min(int(len(label_pred_sorted) * p / 100), len(label_pred_sorted) - 1)]
    cpu_idx = label_pred < pct_thr
    gpu_idx = label_pred >= pct_thr
    cpu_recall_cnt = test_number_recall[cpu_idx]
    overall_recall = (np.sum(cpu_recall_cnt) + 1000 * np.sum(gpu_idx)) / len(test_label_regression)
    # print(f'{p}%->{pct_thr:.2f} | cpu workload: {np.sum(cpu_idx)} | \
    #     gpu workload: {np.sum(gpu_idx)} | \
    #     overall recall: {overall_recall:6f}')
    scores += overall_recall

scores /= (100 / step)
print(f'avg scores: {scores:.2f}')
    # print(f'{p}%->bruteforce | predict recall: {recalls[p]:4f} | overall recall: {overall_recall:6f}')

print(f'test_comps: {np.mean(test_comps):.2f}')
print(f'test_comps: {np.sum(test_label_regression)} / {len(test_label_regression)}')
index = np.arange(len(test_label_regression))
print("*" * 100)

hnns_cpu_idx, hnns_gpu_idx = label_pred < pct50, label_pred >= pct50
hnns_cpu_comps, hnns_gpu_comps = test_comps[hnns_cpu_idx], test_comps[hnns_gpu_idx]
hnns_miss = test_label_regression[hnns_cpu_idx]
print(f'hnns_cpu_comps: {np.mean(hnns_cpu_comps):.2f} | count: {len(hnns_cpu_comps)}')
print(f'hnns_gpu_comps: {np.mean(hnns_gpu_comps):.2f} | count: {len(hnns_gpu_comps)}')
print(f'hnns_miss: {np.sum(hnns_miss)}')
print()

random_cpu_idx, random_gpu_idx = train_test_split(index, test_size = 0.5, random_state = 42)
random_cpu_comps, random_gpu_comps = test_comps[random_cpu_idx], test_comps[random_gpu_idx]
random_miss = test_label_regression[random_cpu_idx]
print(f'random_cpu_comps: {np.mean(random_cpu_comps):.2f} | count: {len(random_cpu_comps)}')
print(f'random_gpu_comps: {np.mean(random_gpu_comps):.2f} | count: {len(random_gpu_comps)}')
print(f'random_miss: {np.sum(random_miss)}')
print()
