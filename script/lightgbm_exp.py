# conda activate /home/zhengweiguo/miniconda3/envs/lcj_bert
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from binary_io import *
import json
import os

dataset = 'imagenet'
dataset = 'wikipedia'
dataset = 'gist1m'
dataset = 'datacomp-image'
dataset = 'deep100m'
config = json.loads(open('config.json').read())
M, efs = config[dataset]["M"], config[dataset]["efs"]
dim = config[dataset]["dim"]
efc = 1000
ck_ts = 2000
k = 1000
threshold = 980

# data_prefix = '/data/disk1/liuchengjun/HNNS/sample/ratio/'
data_prefix = '/data/disk1/liuchengjun/HNNS/sample/'
checkpoint_prefix = '/data/disk1/liuchengjun/HNNS/checkpoint/'
prefix = f'{dataset}.M_{M}.efc_{efc}.efs_{efs}.ck_ts_{ck_ts}.ncheck_100.recall@{k}'
checkpoint_path = f'{checkpoint_prefix}{prefix}.thr_{threshold}.txt'

train_feature = fvecs_read(f'{data_prefix}{prefix}.train_feats_nn.fvecs')
train_number_recall = ivecs_read(f'{data_prefix}{prefix}.train_label.ivecs')[:, 0]
train_label = train_number_recall < threshold
train_comps = ivecs_read(f'{data_prefix}{prefix}.train_label.ivecs')[:, 1]
test_feature = fvecs_read(f'{data_prefix}{prefix}.test_feats_nn.fvecs')
test_number_recall = ivecs_read(f'{data_prefix}{prefix}.test_label.ivecs')[:, 0]
test_label = test_number_recall < threshold
test_comps = ivecs_read(f'{data_prefix}{prefix}.test_label.ivecs')[:, 1]

# print(len(train_feature), len(train_label))
print(np.sum(test_label), np.sum(train_label))
# print(len(train_feature[0]), train_feature[0][-101:])

offset = 0
feat_query = train_feature[:, offset: offset + dim]
offset += dim
feat_dist = train_feature[:, offset: offset + 100]
offset += 100
feat_update = train_feature[:, offset: offset + 3]
offset += 3
assert offset == len(train_feature[0])

query_cols = [f"query_{i}" for i in range(dim)]
dist_cols = [f"dist_{i}" for i in range(100)]
update_cols = [f"update_{i}" for i in range(3)]

df_query = pd.DataFrame(feat_query, columns=query_cols)
df_dist = pd.DataFrame(feat_dist, columns=dist_cols)
df_update = pd.DataFrame(feat_update, columns=update_cols)

df = pd.concat([df_query, df_dist, df_update], axis = 1)

##################################################  ##################################################

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'learning_rate': 0.05,
    'num_boost_round': 3000,
    'verbose': 1,
    "num_threads": 96
}

##################################################  ##################################################

if os.path.exists(checkpoint_path):
    print(f'[Checkpoint] {checkpoint_path} exist!')
    gbm = lgb.Booster(model_file = checkpoint_path)
    print('[Checkpoint] Loaded!')
else:
    print(f'[Checkpoint] {checkpoint_path} not exist!')
    print('[Checkpoint] Training!')
    gbm = lgb.train(params, lgb.Dataset(df.values, label=train_label))
    gbm.save_model(checkpoint_path)
    print('[Checkpoint] Done!')

train_pred = gbm.predict(train_feature)
sorted_train_pred = np.sort(train_pred)
# pct50 = np.percentile(train_pred, 0.5)
# pct50 = sorted_train_pred[int(len(sorted_train_pred) * 0.5)]
pct50 = 0.30
print(f'pct50 threshold: {pct50}')

importance = gbm.feature_importance()
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
label_pred = gbm.predict(test_feature)
end = time.time()

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
print(f"Area Under the Curve (AUC): {auc:.4f}")

total_true_label = np.sum(test_label)
step = 2
for p in range(0, 100 + step, step):
    fail = total_true_label * (1 - recalls[p])
    success = len(test_label) - fail
    overall_recall = (success * 1.000 + fail * 0.997) / len(test_label)
    # print(f'{p}%->bruteforce | predict recall: {recalls[p]:4f} | overall recall: {overall_recall:6f}')

plt.figure(figsize=(10, 8))
plt.plot(percentages, recalls)
plt.xlabel("Top percentage of positive examples selected (GPU Budget)")
plt.ylabel("Recall")
fig_path = f'{prefix}.thr_{threshold}'
plt.title(fig_path)
plt.savefig(fig_path + '.png', dpi=300)
plt.show()

train_number_recall_mn = np.min(train_number_recall)
hist, bins = np.histogram(train_number_recall, bins = range(0, 1000 + 1))
plt.bar(bins[: -1], hist, width = 1)
plt.xlabel('train_number_recall / 1000 nearest neighbors')
plt.xlim([train_number_recall_mn - 10, 1000])
plt.ylim(0.1)
plt.yscale('log')
plt.ylabel('Frequency')
plt.title(f'{prefix}\nFrequency Distribution of train_number_recall')
print(f'train_number_recall_mn: {train_number_recall_mn}')
fig_path = f'{prefix}.recall_hist'
plt.savefig(fig_path + '.png', dpi=300)
plt.show()

plt.figure(figsize=(10, 8))
# plt.scatter(x = train_number_recall, y = train_comps, s=1)
train_comps_filtered = train_comps[train_number_recall >= 950]
train_number_recall_filtered = train_number_recall[train_number_recall >= 950]
origin_heatmap, xedges, yedges = np.histogram2d(train_number_recall_filtered, train_comps_filtered, bins=50)
heatmap = np.log10(origin_heatmap + 1)

hm = plt.imshow(heatmap.T, origin = 'lower', extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect = 'auto', cmap = 'viridis')

plt.xlabel(f'number_recall / 1000 nearest neighbors\n{len(train_number_recall)} points')
plt.ylabel('NDC')
# plt.xlim([990, 1000])
ax = plt.gca()
ax.invert_xaxis()
plt.title(f'{prefix}\nHeatmap of number_recall vs NDC')
cbar = plt.colorbar(hm)
cbar.set_label('Number of points (log10)')

fig_path = f'{prefix}.Comp_R'
plt.savefig(fig_path + '.png', dpi=300)
plt.show()
# for R in range(950, 1001):
#     print(f'avg comparison for {R/1000}: {np.mean(train_comps[train_number_recall == R]):.2f}')

# pearson coef r
r, p = pearsonr(train_number_recall, train_comps)
print(f'pearsonr: {r:.2f} | p: {p:.2f}')

# roc_auc score, roc_auc_score
from sklearn.metrics import roc_auc_score
roc_auc_score = roc_auc_score(test_label, label_pred)
print(f'roc_auc_score: {roc_auc_score:.2f}')

##################################################  ##################################################

print(f'test_comps: {np.mean(test_comps):.2f}')
print(f'test_comps: {np.sum(test_label)} / {len(test_label)}')
index = np.arange(len(test_label))
print("*" * 100)

hnns_cpu_idx, hnns_gpu_idx = label_pred < pct50, label_pred >= pct50
hnns_cpu_comps, hnns_gpu_comps = test_comps[hnns_cpu_idx], test_comps[hnns_gpu_idx]
hnns_miss = test_label[hnns_cpu_idx]
ivecs_write(f'{checkpoint_prefix}{prefix}.thr_{threshold}.hnns_cpu_idx.ivecs', hnns_cpu_idx.reshape(-1, 1))
ivecs_write(f'{checkpoint_prefix}{prefix}.thr_{threshold}.hnns_gpu_idx.ivecs', hnns_gpu_idx.reshape(-1, 1))
print(f'hnns_cpu_comps: {np.mean(hnns_cpu_comps):.2f} | count: {len(hnns_cpu_comps)}')
print(f'hnns_gpu_comps: {np.mean(hnns_gpu_comps):.2f} | count: {len(hnns_gpu_comps)}')
print(f'hnns_miss: {np.sum(hnns_miss)}')
print()

random_cpu_idx, random_gpu_idx = train_test_split(index, test_size = 0.5, random_state = 42)
random_cpu_comps, random_gpu_comps = test_comps[random_cpu_idx], test_comps[random_gpu_idx]
random_miss = test_label[random_cpu_idx]
ivecs_write(f'{checkpoint_prefix}{prefix}.thr_{threshold}.random_cpu_idx.ivecs', random_cpu_idx.reshape(-1, 1))
ivecs_write(f'{checkpoint_prefix}{prefix}.thr_{threshold}.random_gpu_idx.ivecs', random_gpu_idx.reshape(-1, 1))
print(f'random_cpu_comps: {np.mean(random_cpu_comps):.2f} | count: {len(random_cpu_comps)}')
print(f'random_gpu_comps: {np.mean(random_gpu_comps):.2f} | count: {len(random_gpu_comps)}')
print(f'random_miss: {np.sum(random_miss)}')
print()
