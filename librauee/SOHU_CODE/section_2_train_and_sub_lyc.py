import warnings
import os
import re
import gc
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, minmax_scale
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import auc, roc_auc_score

import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
import seaborn as sns
%matplotlib inline
%config InlineBackend.figure_format='retina' # 主题

train = pd.read_csv('Sohu2022_data/train-dataset.csv')
train2 = pd.read_csv('Sohu2022_data/rec-train-dataset.csv')
train = pd.concat([train, train2]).reset_index()
del train2
test = pd.read_csv('Sohu2022_data/rec-test-dataset.csv')
print("train_data.shape",train.shape)
print("test_data.shape",test.shape)
train.head()

data = pd.concat([train, test]).reset_index(drop=True)
del train, test
gc.collect()

data.sort_values('logTs', inplace=True)

data['log_pv_diff'] = data.groupby(['pvId'])['logTs'].diff()
df_temp = data.groupby(['pvId'])['log_pv_diff'].agg([
    ('day_range_max', 'max'),
    ('day_range_min', 'min'),
    ('day_range_mean', 'mean'),
    ('day_range_std', 'std'),
    ('day_range_skew', lambda x: x.skew()),
])
data = pd.merge(data, df_temp, on='pvId', how='left')

data['log_suv_diff'] = data.groupby(['suv'])['logTs'].diff()
df_temp = data.groupby(['suv'])['log_suv_diff'].agg([
    ('suv_range_max', 'max'),
    ('suv_range_min', 'min'),
    ('suv_range_mean', 'mean'),
    ('suv_range_std', 'std'),
    ('suv_range_skew', lambda x: x.skew()),
])
data = pd.merge(data, df_temp, on='suv', how='left')

data['log_itemId_diff'] = data.groupby(['itemId'])['logTs'].diff()
df_temp = data.groupby(['itemId'])['log_itemId_diff'].agg([
    ('itemId_range_max', 'max'),
    ('itemId_range_min', 'min'),
    ('itemId_range_mean', 'mean'),
    ('itemId_range_std', 'std'),
    ('itemId_range_skew', lambda x: x.skew()),
])
data = pd.merge(data, df_temp, on='itemId', how='left')

del df_temp
gc.collect()

res_nlp = pd.read_csv('data/res(1).csv')
ss = res_nlp.groupby(['id'])['pred']
ss = ss.apply(lambda x: [[float(j) for j in i[1:-1].split()] for i in x]).reset_index()

def get_nlp_count(x):
    tmp = {0: 0, 1:0, 2:0, 3:0, 4:0}
    for i in range(len(x)):
        tmp[np.array(x[i]).argmax()] += 1  
    return sorted(tmp.items(), key=lambda x: x[1])[-1][0] if len(x) > 0 else 2

ss['senti'] = ss['pred'].apply(get_nlp_count)

data_seq = data.drop_duplicates(['suv'])
data_seq['userSeq'] = data_seq['userSeq'].fillna('')
data_seq['seq1'] = data_seq['userSeq'].apply(lambda x: [int(i.split(':')[0]) if i.split(':')[0].isdigit() else 0 for i in x.split(';')])
data_seq_exp = data_seq.explode('seq1')

data_seq_exp = data_seq_exp.merge(ss, left_on='seq1', right_on='id', how='left')
sss = data_seq_exp.groupby('suv')['senti'].value_counts()
sss = pd.DataFrame(sss)
sss.columns = ['senti_counts']
sss = sss.pivot_table(index='suv', columns='senti', values=['senti_counts']).reset_index().fillna(0)
sss.columns = ['suv', 'senti_counts_0', 'senti_counts_1', 'senti_counts_2', 'senti_counts_3', 'senti_counts_4']
data = pd.merge(data, sss, on='suv', how='left')
del sss, ss, res_nlp, data_seq_exp
gc.collect()

data['pvId_rank'] = data.groupby(['pvId'])['logTs'].rank()
data['suv_rank'] = data.groupby(['pvId', 'suv'])['logTs'].rank()
data['itemId_rank'] = data.groupby(['pvId', 'itemId'])['logTs'].rank()

data['suv_rank/pvId_rank'] = data['suv_rank'] / data['pvId_rank']
data['itemId_rank/pvId_rank'] = data['itemId_rank'] / data['pvId_rank']

data['suv_rank_pvId_rank'] = data['suv_rank'] - data['pvId_rank']
data['itemId_rank_pvId_rank'] = data['itemId_rank'] - data['pvId_rank']

data['suv_rank/itemId_rank'] = data['suv_rank'] / data['itemId_rank']
data['suv_rank_itemId_rank'] = data['suv_rank'] - data['itemId_rank']

data['prov_city']=data['province']+data['city']
data['device_os']=data['deviceType']+data['osType']
data['opera_browser']=data['operator']+data['browserType']
data['device_os_opera_browser']=data['device_os']+data['opera_browser']
data['suv_pvId']=data['suv']+'_'+data['pvId']
data['prov_city_device_os']=data['prov_city']+data['device_os']
data['prov_city_opera_browser']=data['prov_city']+data['opera_browser']

sparse_features = ['pvId', 'suv', 'itemId', 'operator', 'browserType', 
                   'deviceType', 'osType', 'province', 'city',                   
                   'prov_city', 'device_os', 'opera_browser', 
                   'device_os_opera_browser', 'prov_city_device_os', 'prov_city_opera_browser'
]
dense_features = ['pvId_rank', 'itemId_rank', 'suv_rank', 
                  'suv_rank/pvId_rank', 'itemId_rank/pvId_rank',
                  'suv_rank_pvId_rank', 'itemId_rank_pvId_rank', 'suv_rank/itemId_rank', 'suv_rank_itemId_rank',
                  'senti_counts_0', 'senti_counts_1', 'senti_counts_2', 'senti_counts_3', 'senti_counts_4',
                  'day_range_max', 'day_range_min', 'day_range_mean', 'day_range_std', 'day_range_skew',
                  'suv_range_max', 'suv_range_min', 'suv_range_mean', 'suv_range_std', 'suv_range_skew',
                  'itemId_range_max', 'itemId_range_min', 'itemId_range_mean', 'itemId_range_std', 'itemId_range_skew',
                  'log_pv_diff', 'log_suv_diff', 'log_itemId_diff',
                  'logTs',
                 ] 

target = 'label'

for feat in tqdm(sparse_features):
    lb = LabelEncoder()
    data[feat] = lb.fit_transform(data[feat])

data['logTs'] = pd.to_datetime(data['logTs'],unit='ms')
data['day'] = data['logTs'].dt.day

## 当天曝光
data['day_item_cnt'] = data.groupby(['itemId','day'])['logTs'].transform('count')
data['day_suv_cnt'] = data.groupby(['suv','day'])['logTs'].transform('count')
data['day_suv_itemId_cnt'] = data.groupby(['suv','day','itemId'])['logTs'].transform('count')
data['day_item_nunique'] = data.groupby(['itemId','day'])['logTs'].transform('nunique')
data['day_suv_nunique'] = data.groupby(['suv','day'])['logTs'].transform('nunique')
data['day_suv_itemId_nunique'] = data.groupby(['suv','day','itemId'])['logTs'].transform('nunique')

dense_features.extend(['day', 'day_item_cnt', 'day_suv_cnt', 'day_suv_itemId_cnt',
                      'day_item_nunique', 'day_suv_nunique', 'day_suv_itemId_nunique'
                      ])

for col in tqdm(sparse_features):
    if col != 'pvId':
        data[col + '_pvId_count'] = data.groupby(['pvId', col])['logTs'].transform('count')
        dense_features.append(col + '_pvId_count')
        data[col + '_pvId_nunique'] = data.groupby(['pvId', col])['logTs'].transform('nunique')
        dense_features.append(col + '_pvId_nunique')

# count特征
for col in tqdm(sparse_features):
    data[col + '_count'] = data.groupby(col)['logTs'].transform('count')
    dense_features.append(col + '_count')
    
# count特征
for col in tqdm(['pvId','itemId' ]):
    data[f'group_suv_{col}_nunique'] = \
        data[['suv', col]].groupby('suv')[col].transform('nunique')
    dense_features.append(f'group_suv_{col}_nunique')  
    
# pvId nunique特征
select_cols = ['suv', 'itemId']
for col in tqdm(select_cols):
    data[f'group_pvId_{col}_nunique'] = \
        data[['pvId', col]].groupby('pvId')[col].transform('nunique')
    dense_features.append(f'group_pvId_{col}_nunique')  
    
# itemId nunique特征
select_cols = ['pvId', 'suv', 'operator', 'browserType', 
             'deviceType', 'osType', 'province', 'city']
for col in tqdm(select_cols):
    data[f'group_itemId_{col}_nunique'] = \
        data[['itemId', col]].groupby('itemId')[col].transform('nunique')
    dense_features.append(f'group_itemId_{col}_nunique') 

data['userSeq']=data['userSeq'].fillna('')
data['userSeq_len']=data['userSeq'].apply(lambda x:len(x.split(';')))
dense_features.append(f'userSeq_len') 

for f in tqdm(sparse_features):
    if f not in ['pvId', 'suv', 'itemId']:
        data[f'{f}_rank'] = data.groupby(['pvId', f])['logTs'].rank()
        dense_features.append(f'{f}_rank')
        
        
data[dense_features] = data[dense_features].fillna(-1)
mms = MinMaxScaler(feature_range=(0, 1))
for f in tqdm(dense_features):
    data[f] = mms.fit_transform(data[f].values.reshape(-1, 1))

train = data[~data['label'].isna()]
test = data[data['label'].isna()]

ss = train['logTs'].quantile(q=0.8)
valid = train[train['logTs'] > ss].reset_index(drop=True)
train = train[train['logTs'] <= ss].reset_index(drop=True)
del data
gc.collect()

from collections import defaultdict
import numpy as np
import pandas as pd
from numba import njit
from scipy.stats import rankdata


@njit
def _auc(actual, pred_ranks):
    n_pos = np.sum(actual)
    n_neg = len(actual) - n_pos
    return (np.sum(pred_ranks[actual == 1]) - n_pos*(n_pos+1)/2) / (n_pos*n_neg)


def fast_auc(actual, predicted):
    # https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/208031
    pred_ranks = rankdata(predicted)
    return _auc(actual, pred_ranks)


def uAUC(labels, preds, user_id_list):
    """Calculate user AUC"""
    user_pred = defaultdict(lambda: [])
    user_truth = defaultdict(lambda: [])
    for idx, truth in enumerate(labels):
        user_id = user_id_list[idx]
        pred = preds[idx]
        truth = labels[idx]
        user_pred[user_id].append(pred)
        user_truth[user_id].append(truth)

    user_flag = defaultdict(lambda: False)
    for user_id in set(user_id_list):
        truths = user_truth[user_id]
        flag = False
        # 若全是正样本或全是负样本，则flag为False
        for i in range(len(truths) - 1):
            if truths[i] != truths[i + 1]:
                flag = True
                break
        user_flag[user_id] = flag

    total_auc = 0.0
    size = 0.0
    for user_id in user_flag:
        if user_flag[user_id]:
            auc = fast_auc(np.asarray(user_truth[user_id]), np.asarray(user_pred[user_id]))
            total_auc += auc
            size += 1.0
    user_auc = float(total_auc)/size
    return user_auc

from collections import Counter, defaultdict

import numpy as np
from sklearn.model_selection._split import _BaseKFold
from sklearn.utils.validation import check_random_state


class StratifiedGroupKFold(_BaseKFold):

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    # Implementation based on this kaggle kernel:
    # https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
    def _iter_test_indices(self, X, y, groups):
        labels_num = np.max(y) + 1
        y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
        y_distr = Counter()
        for label, group in zip(y, groups):
            y_counts_per_group[group][label] += 1
            y_distr[label] += 1

        y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
        groups_per_fold = defaultdict(set)

        groups_and_y_counts = list(y_counts_per_group.items())
        rng = check_random_state(self.random_state)
        if self.shuffle:
            rng.shuffle(groups_and_y_counts)

        for group, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
            best_fold = None
            min_eval = None
            for i in range(self.n_splits):
                y_counts_per_fold[i] += y_counts
                std_per_label = []
                for label in range(labels_num):
                    std_per_label.append(
                        np.std([y_counts_per_fold[j][label] / y_distr[label] for j in range(self.n_splits)])
                    )
                y_counts_per_fold[i] -= y_counts
                fold_eval = np.mean(std_per_label)
                if min_eval is None or fold_eval < min_eval:
                    min_eval = fold_eval
                    best_fold = i
            y_counts_per_fold[best_fold] += y_counts
            groups_per_fold[best_fold].add(group)

        for i in range(self.n_splits):
            test_indices = [idx for idx, group in enumerate(groups) if group in groups_per_fold[i]]
            yield test_indices

import gc
import logging
import os
import pickle
from logging import getLogger, INFO, FileHandler, StreamHandler

import numpy as np
import pandas as pd
from catboost import Pool, CatBoostClassifier

feas=sparse_features+dense_features

gskf = StratifiedGroupKFold(n_splits=5)
oof = np.zeros((len(train)))
pred_cat = []

for fold_, (train_idxs, test_idxs) in enumerate(gskf.split(train['label'], train['label'].astype(int), train['pvId'].astype(int))):
    print(fold_)
    train_dataset = Pool(train.iloc[train_idxs][feas], train.iloc[train_idxs]['label'].astype(int))
    eval_dataset = Pool(train.iloc[test_idxs][feas], train.iloc[test_idxs]['label'].astype(int))

    train_model = CatBoostClassifier(iterations=1300, depth=5, learning_rate=0.05, loss_function='Logloss',
                                         logging_level='Verbose', eval_metric='Logloss', task_type="GPU")
    train_model.fit(train_dataset, eval_set=eval_dataset, early_stopping_rounds=100, verbose=40)
    train_model.save_model(f'model/model_{fold_}.json')
    oof[test_idxs] = train_model.predict_proba(eval_dataset)[:, 1]
    pred_cat.append(train_model.predict_proba(test[feas])[:, 1])   

userid_list = train['pvId'].astype(str).tolist()
train['pred_prob'] = oof

print("valid AUC:", round(roc_auc_score(train[target].values, oof), 4))
print("valid gAUC:", round(uAUC(list(train['label'].values), list(train['pred_prob'].values), userid_list), 4))

sub = test[['testSampleId']]
sub.columns = ['Id']
sub['result'] = np.mean(pred_cat, axis=0)
sub.to_csv('section2_v_f05.txt', sep='\t', index=False)
sub.head()

