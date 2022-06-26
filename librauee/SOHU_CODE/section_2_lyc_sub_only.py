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
from collections import Counter, defaultdict

import numpy as np
from sklearn.model_selection._split import _BaseKFold
from sklearn.utils.validation import check_random_state

import gc
import logging
import os
import pickle
from logging import getLogger, INFO, FileHandler, StreamHandler

import numpy as np
import pandas as pd
from catboost import Pool, CatBoostClassifier


test = pd.read_pickle('data/test.pkl')
feas = ['pvId',
 'suv',
 'itemId',
 'operator',
 'browserType',
 'deviceType',
 'osType',
 'province',
 'city',
 'prov_city',
 'device_os',
 'opera_browser',
 'device_os_opera_browser',
 'prov_city_device_os',
 'prov_city_opera_browser',
 'pvId_rank',
 'itemId_rank',
 'suv_rank',
 'suv_rank/pvId_rank',
 'itemId_rank/pvId_rank',
 'suv_rank_pvId_rank',
 'itemId_rank_pvId_rank',
 'suv_rank/itemId_rank',
 'suv_rank_itemId_rank',
 'senti_counts_0',
 'senti_counts_1',
 'senti_counts_2',
 'senti_counts_3',
 'senti_counts_4',
 'day_range_max',
 'day_range_min',
 'day_range_mean',
 'day_range_std',
 'day_range_skew',
 'suv_range_max',
 'suv_range_min',
 'suv_range_mean',
 'suv_range_std',
 'suv_range_skew',
 'itemId_range_max',
 'itemId_range_min',
 'itemId_range_mean',
 'itemId_range_std',
 'itemId_range_skew',
 'log_pv_diff',
 'log_suv_diff',
 'log_itemId_diff',
 'logTs',
 'day',
 'day_item_cnt',
 'day_suv_cnt',
 'day_suv_itemId_cnt',
 'day_item_nunique',
 'day_suv_nunique',
 'day_suv_itemId_nunique',
 'suv_pvId_count',
 'suv_pvId_nunique',
 'itemId_pvId_count',
 'itemId_pvId_nunique',
 'operator_pvId_count',
 'operator_pvId_nunique',
 'browserType_pvId_count',
 'browserType_pvId_nunique',
 'deviceType_pvId_count',
 'deviceType_pvId_nunique',
 'osType_pvId_count',
 'osType_pvId_nunique',
 'province_pvId_count',
 'province_pvId_nunique',
 'city_pvId_count',
 'city_pvId_nunique',
 'prov_city_pvId_count',
 'prov_city_pvId_nunique',
 'device_os_pvId_count',
 'device_os_pvId_nunique',
 'opera_browser_pvId_count',
 'opera_browser_pvId_nunique',
 'device_os_opera_browser_pvId_count',
 'device_os_opera_browser_pvId_nunique',
 'prov_city_device_os_pvId_count',
 'prov_city_device_os_pvId_nunique',
 'prov_city_opera_browser_pvId_count',
 'prov_city_opera_browser_pvId_nunique',
 'pvId_count',
 'suv_count',
 'itemId_count',
 'operator_count',
 'browserType_count',
 'deviceType_count',
 'osType_count',
 'province_count',
 'city_count',
 'prov_city_count',
 'device_os_count',
 'opera_browser_count',
 'device_os_opera_browser_count',
 'prov_city_device_os_count',
 'prov_city_opera_browser_count',
 'group_suv_pvId_nunique',
 'group_suv_itemId_nunique',
 'group_pvId_suv_nunique',
 'group_pvId_itemId_nunique',
 'group_itemId_pvId_nunique',
 'group_itemId_suv_nunique',
 'group_itemId_operator_nunique',
 'group_itemId_browserType_nunique',
 'group_itemId_deviceType_nunique',
 'group_itemId_osType_nunique',
 'group_itemId_province_nunique',
 'group_itemId_city_nunique',
 'userSeq_len',
 'operator_rank',
 'browserType_rank',
 'deviceType_rank',
 'osType_rank',
 'province_rank',
 'city_rank',
 'prov_city_rank',
 'device_os_rank',
 'opera_browser_rank',
 'device_os_opera_browser_rank',
 'prov_city_device_os_rank',
 'prov_city_opera_browser_rank']

pred_cat = []

for fold_ in range(5):
    print(fold_)
    train_dataset = Pool(train.iloc[train_idxs][feas], train.iloc[train_idxs]['label'].astype(int))
    eval_dataset = Pool(train.iloc[test_idxs][feas], train.iloc[test_idxs]['label'].astype(int))

    train_model = CatBoostClassifier(iterations=1300, depth=5, learning_rate=0.05, loss_function='Logloss',
                                         logging_level='Verbose', eval_metric='Logloss', task_type="GPU")
    train_model.load_model(f'model/model_{fold_}.json')
    pred_cat.append(train_model.predict_proba(test[feas])[:, 1])

sub = test[['testSampleId']]
sub.columns = ['Id']
sub['result'] = np.mean(pred_cat, axis=0)
sub.to_csv('section2_v_f05.txt', sep='\t', index=False)
sub.head()

