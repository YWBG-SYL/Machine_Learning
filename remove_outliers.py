#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：广工_公选课_捡漏.py 
@File    ：t1.py
@IDE     ：PyCharm 
@Author  ：原味不改
@Date    ：2024/10/28 11:06 
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from my_svm import SVM
import matplotlib
matplotlib.use('TkAgg')
from scipy.io import loadmat
from scipy.stats import gaussian_kde
from typing import Tuple, Optional, Union

def remove_outliers( X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据标签对数据点密度进行估计，去除某一类别中的孤立点。
    """
    X_filtered, y_filtered = [], []

    # 分别处理每个类别的数据
    for label in np.unique(y):
        X_label = X[y == label]
        kde = gaussian_kde(X_label.T)
        densities = kde(X_label.T)

        # 根据密度的百分位数作为阈值来过滤孤立点
        threshold = np.percentile(densities, 2)  # 保留密度最高的 (100-self.kde_threshold)% 的点
        mask = densities > threshold

        X_filtered.extend(X_label[mask])
        y_filtered.extend(y[y == label][mask])

    return np.array(X_filtered), np.array(y_filtered)

raw_data = loadmat('./data/ex6data1.mat')

data = pd.DataFrame(raw_data['X'], columns=['X1', 'X2'])
data['y'] = raw_data['y']
print(data.shape[0])

positive = data[data['y'].isin([1])]  #isin（）为筛选函数，令positive为数据集中为1的数组
negative = data[data['y'].isin([0])]  #isin（）为筛选函数，令negative为数据集中为0的数组

X=raw_data['X']
y = raw_data['y'].flatten()
X_f,y_f=remove_outliers(X,y)
print(X_f.shape[0])

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
ax[0].scatter(positive['X1'], positive['X2'], s=50, marker='x', label='Positive')
ax[0].scatter(negative['X1'], negative['X2'], s=50, marker='o', label='Negative')
ax[0].legend() #图例显示函数
ax[1].scatter(X_f[:,0], X_f[:,1], s=50, marker='o',label='other')
plt.show()







