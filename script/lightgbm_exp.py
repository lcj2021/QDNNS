# conda activate /home/zhengweiguo/miniconda3/envs/lcj_bert
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from binary_io import *
import os

dataset = 'gist1m'
dataset = 'imagenet'
dataset = 'wikipedia'
M = 128
efc = 1000
efs = 3000
ck_ts = 2000
k = 1000

data_prefix = '/data/guohaoran/HNNS/sample/'
checkpoint_prefix = '/data/guohaoran/HNNS/checkpoint/'
prefix = f'{dataset}.M_{M}.efc_{efc}.efs_{efs}.ck_ts_{ck_ts}.ncheck_100.recall@{k}'
checkpoint_path = f'{checkpoint_prefix}{prefix}.txt'

train_feature = fvecs_read(f'{data_prefix}{prefix}.train_feats_nn.fvecs')
train_label = ivecs_read(f'{data_prefix}{prefix}.train_label.ivecs')[:, 0]
train_comps = ivecs_read(f'{data_prefix}{prefix}.train_label.ivecs')[:, 1]
test_feature = fvecs_read(f'{data_prefix}{prefix}.test_feats_nn.fvecs')
test_label = ivecs_read(f'{data_prefix}{prefix}.test_label.ivecs')[:, 0]
test_comps = ivecs_read(f'{data_prefix}{prefix}.test_label.ivecs')[:, 1]

# train_feature = train_feature[:, -100:]
# test_feature = test_feature[:, -100:]
print(len(train_feature), len(train_label))
print(np.sum(test_label), np.sum(train_label))
print(len(train_feature[0]), train_feature[0][-101:])
# train_feature = train_feature[:, :-3]
# test_feature = test_feature[:, :-3]


##################################################  ##################################################

params = {
    'boosting_type': 'gbdt',    # 使用 GBDT 算法
    'objective': 'binary',  # 二分类任务
    'metric': 'binary_logloss',            # 使用 binary_logloss 作为评估指标
    'learning_rate': 0.05,       # 学习率
    'num_boost_round': 3000,     # 提升迭代的轮数
    'verbose': 1,               # 控制处理信息的详细程度
    'num_threads': 128           # 设置线程数
}

##################################################  ##################################################

if os.path.exists(checkpoint_path):
    print(f'[Checkpoint] {checkpoint_path} exist!')
    gbm = lgb.Booster(model_file = checkpoint_path)
    print('[Checkpoint] Loaded!')
else:
    print(f'[Checkpoint] {checkpoint_path} not exist!')
    print('[Checkpoint] Training!')
    gbm = lgb.train(params, lgb.Dataset(train_feature, label=train_label))
    gbm.save_model(checkpoint_path)
    print('[Checkpoint] Done!')

train_pred = gbm.predict(train_feature)
pct50 = np.median(train_pred)
print(f'pct50 threshold: {pct50}')

##################################################  ##################################################

from sklearn.metrics import recall_score
import time
start = time.time()
label_pred = gbm.predict(test_feature)
end = time.time()

import numpy as np
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid as trapz

def recall_curve(test_label, label_pred):
    n = len(label_pred)
    recalls = []
    percentages = np.arange(0, 1.01, 0.01)  # 从0%到100%，步长为1%
    for p in percentages:
        threshold = np.percentile(label_pred, 100 * (1 - p))
        if p==0: threshold += 1
        new_label_pred = np.where(label_pred >= threshold, 1, 0)
        recall = recall_score(test_label, new_label_pred)
        recalls.append(recall)
    return percentages, recalls

percentages, recalls = recall_curve(test_label, label_pred)
auc = trapz(recalls, percentages)
print("Area Under the Curve (AUC):", auc)

total_true_label = np.sum(test_label)
step = 2
for p in range(0, 100 + step, step):
    fail = total_true_label * (1 - recalls[p])
    success = len(test_label) - fail
    overall_recall = (success * 1.000 + fail * 0.997) / len(test_label)
    print(f'{p}%->bruteforce | predict recall: {recalls[p]:4f} | overall recall: {overall_recall:6f}')

plt.plot(percentages, recalls)
plt.xlabel("Top percentage of positive examples selected (GPU Budget)")
plt.ylabel("Recall")

plt.title(f'{dataset}.M_{M}.efc_{efc}.efs_{efs}.ck_ts_{ck_ts}.ncheck_100.recall@{k}')
plt.savefig(f'{dataset}.M_{M}.efc_{efc}.efs_{efs}.ck_ts_{ck_ts}.ncheck_100.recall@{k}.png')
plt.show()

# recall = recall_score(test_label, label_pred.round())
# print("Recall:", recall)

##################################################  ##################################################

print(f'test_comps: {np.mean(test_comps)}')
print(f'test_comps: {np.sum(test_label)}')
print("*" * 100)

hnns_cpu_comps = test_comps[label_pred < pct50]
hnns_gpu_comps = test_comps[label_pred >= pct50]
hnns_miss = test_label[label_pred < pct50]
print(f'hnns_cpu_comps: {np.mean(hnns_cpu_comps)}')
print(f'hnns_gpu_comps: {np.mean(hnns_gpu_comps)}')
print(f'hnns_miss: {np.sum(hnns_miss)}')
print()

random_indices = np.arange(len(test_label))
random_cpu_idx, random_gpu_idx = train_test_split(random_indices, test_size = 0.5, random_state = 42)
random_cpu_comps, random_gpu_comps = test_comps[random_cpu_idx], test_comps[random_gpu_idx]
random_miss = test_label[random_cpu_idx]
print(f'random_cpu_comps: {np.mean(random_cpu_comps)}')
print(f'random_gpu_comps: {np.mean(random_gpu_comps)}')
print(f'random_miss: {np.sum(random_miss)}')
print()