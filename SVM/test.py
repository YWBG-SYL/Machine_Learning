#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@File    ：svm_test.py
@IDE     ：PyCharm 
@Author  ：原味不改
@Date    ：2024/10/27 16:54 
'''

import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from my_svm import SVM
import matplotlib
matplotlib.use('TkAgg')
from scipy.io import loadmat

# 加载数据
df_data = loadmat('./data/ex6data1.mat')
data = pd.DataFrame(df_data['X'], columns=['X1', 'X2'])
data['y'] = df_data['y']

X = df_data['X']  # 形状为 (51, 2)
y = df_data['y'].flatten()  # 将 y 展平为一维数组，形状为 (51,)

svm = SVM(max_iter=100,kernel='linear', C=1,random_seed=42,kde_threshold=2)
svm.fit(X, y)

# 获取权重向量和偏置项
w, b = svm.get_params()


print("Weight vector:", w)
print("Bias term:", b)


# 绘制决策边界
def plot_svm_decision_boundary(X, y, w, b):
    # 计算决策边界
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # 创建网格
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # 计算每个点的决策函数值
    Z = np.dot(np.c_[xx.ravel(), yy.ravel()], w) + b
    Z = Z.reshape(xx.shape)

    # 绘制决策边界
    plt.contour(xx, yy, Z, levels=[0], colors='k', linewidths=2)

    # 绘制支持向量
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1', marker='o')
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 0', marker='x')

    plt.title("SVM Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True)
    plt.show()


# 调用绘制函数
plot_svm_decision_boundary(X, y, w, b)








