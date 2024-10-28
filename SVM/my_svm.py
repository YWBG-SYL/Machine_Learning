#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@File    ：my_svm.py
@IDE     ：PyCharm 
@Author  ：原味不改
@Date    ：2024/10/27 19:56 
'''

import numpy as np
from typing import Tuple, Optional, Union
from scipy.stats import gaussian_kde


class SVM:
    def __init__(self, max_iter=100, kernel='linear', C=1.0, tol=1e-4,gamma=1.0,random_seed: int = None,kde_threshold=2):
        self.max_iter = max_iter
        self.kernel = kernel
        self.C = C
        self.tol = tol
        self.alpha = None
        self.b = 0
        self.X = None
        self.y = None
        self.w = None
        self.gamma = gamma
        self.K=None#定义核矩阵
        self.kde_threshold = kde_threshold  # KDE 密度阈值
        # 声明 error_cache 为与样本数相同大小的数组，初始值为 None
        self.error_cache = None
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)  # 设置随机种子

    #定义核行数
    def _kernel_function(self, x1: np.ndarray, x2: np.ndarray) -> float:
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'rbf':
            return np.exp(-self.gamma * np.sum((x1 - x2) ** 2))
        else:
            raise ValueError('未声明核函数')
    #计算核矩阵
    def _get_kernel_matrix(self) -> np.ndarray:
        n_samples = self.X.shape[0]#此处的行数代表样本数
        K = np.zeros((n_samples, n_samples))#用0初始化核矩阵
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self._kernel_function(self.X[i], self.X[j])
        return K

    #计算单个样本的决策值
    def _decision_function_single(self, x: np.ndarray) -> float:
        result = 0.0
        for i in range(len(self.alpha)):
            if self.alpha[i] > 1e-5:  # 只考虑支持向量
                result += self.alpha[i] * self.y[i] * self._kernel_function(self.X[i], x)
        return result + self.b
    #多样本决策值计算
    def _decision_function(self, X: np.ndarray) -> Union[float, np.ndarray]:
        # 处理单个样本的情况
        if X.ndim == 1:
            return self._decision_function_single(X)
        # 处理多个样本的情况
        decision = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            decision[i] = self._decision_function_single(X[i])
        return decision

    def _update_alpha_pair(self, i: int, j: int, K: np.ndarray) -> bool:
        if i == j:
            return False
        alpha_i_old = self.alpha[i].copy()
        alpha_j_old = self.alpha[j].copy()
        y_i, y_j = self.y[i], self.y[j]

        # 计算上下界
        if y_i != y_j:
            L = max(0, alpha_j_old - alpha_i_old)
            H = min(self.C, self.C + alpha_j_old - alpha_i_old)
        else:
            L = max(0, alpha_i_old + alpha_j_old - self.C)
            H = min(self.C, alpha_i_old + alpha_j_old)

        if L == H:
            return False

        # 计算 eta
        eta = 2 * K[i, j] - K[i, i] - K[j, j]
        if eta >= -1e-10:
            return False

        # 计算 E_i 和 E_j，并缓存
        E_i = self._decision_function_single(self.X[i]) - y_i
        E_j = self._decision_function_single(self.X[j]) - y_j

        # 更新 alpha_j
        #self.alpha[j] = np.clip(alpha_j_old - y_j * (E_i - E_j) / eta, L + self.tol, H - self.tol)
        self.alpha[j] = np.clip(alpha_j_old - y_j * (E_i - E_j) / eta, L, H)

        # 检查更新是否显著
        if abs(self.alpha[j] - alpha_j_old) < self.tol:
            return False

        # 更新 alpha_i
        self.alpha[i] = alpha_i_old + y_i * y_j * (alpha_j_old - self.alpha[j])

        # 更新 b
        def _compute_b(b_old, E, y, delta_alpha, K_ii, y_j, delta_alpha_j, K_ij):
            return b_old - E - y * delta_alpha * K_ii - y_j * delta_alpha_j * K_ij

        b1 = _compute_b(self.b, E_i, y_i, self.alpha[i] - alpha_i_old, K[i, i], y_j, self.alpha[j] - alpha_j_old,
                        K[i, j])
        b2 = _compute_b(self.b, E_j, y_i, self.alpha[i] - alpha_i_old, K[i, j], y_j, self.alpha[j] - alpha_j_old,
                        K[j, j])
        # if 0 < self.alpha[i] < self.C:
        #     self.b = b1
        # elif 0 < self.alpha[j] < self.C:
        #     self.b = b2
        # else:
        #     self.b = (b1 + b2) / 2

        if 0 < self.alpha[i] < self.C:
            self.b = b1
        elif 0 < self.alpha[j] < self.C:
            self.b = b2
        else:
            # 确保 w 已正确计算
            if self.w is None:
                # 计算 w，假设 w 的计算是基于 alpha 和支持向量
                self.w = np.sum(self.alpha[i] * self.y[i] * self.X[i] for i in range(len(self.alpha)))
            # 改为仅在所有支持向量上平均计算b
            support_vectors = (self.alpha > 0) & (self.alpha < self.C)
            if np.any(support_vectors):
                self.b = np.mean([self.y[k] - np.dot(self.w, self.X[k]) for k in np.where(support_vectors)[0]])

        #print(self.b)
        for k in range(len(self.y)):
            if 0 < self.alpha[k] < self.C:
                self.error_cache[k] = self._calc_error(k)
        return True



    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVM':
        # 输入数据检查
        X_f,y_f=self._remove_outliers(X,y)
        assert X_f.shape[0] == y_f.shape[0], "样本数量与标签数量不一致"
        assert len(X_f.shape) == 2, "X 应该是一个二维数组"
        self.X = X_f
        self.y = y_f
        n_samples = X_f.shape[0]
        self.alpha = np.zeros(n_samples)
        self.b = 0.0
        #self.error_cache = np.full(n_samples, None, dtype=object)
        # 计算核矩阵
        K = self._get_kernel_matrix()
        # SMO算法主循环
        for _ in range(self.max_iter):
            alpha_changed = 0
            for i in range(n_samples):
                # 选择第二个 alpha
                j, success = self._select_alpha_pair(i)
                if not success:
                    continue
                # 更新选中的 alpha 对
                if self._update_alpha_pair(i, j, K):
                    alpha_changed += 1
            # 检查是否收敛
            if alpha_changed == 0:
                break
        # 计算权重向量（仅对线性核）
        if self.kernel == 'linear':
            self._calculate_w()
        return self


    #计算权重向量
    def _calculate_w(self):
        if self.kernel != 'linear':
            raise ValueError("只能获取线性核的权重向量")
        n_features = self.X.shape[1]
        self.w = np.zeros(n_features)
        for i in range(len(self.alpha)):
            self.w += self.alpha[i] * self.y[i] * self.X[i]

    def get_support_vectors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        support_vector_indices = self.alpha > self.tol
        return (self.X[support_vector_indices],
                self.y[support_vector_indices],
                self.alpha[support_vector_indices])
    #预测结果
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.sign(self._decision_function(X))
    #评估准确率
    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        # 使用模型的 predict 方法生成预测标签
        y_pred = self.predict(X_test)
        # 计算预测值与真实标签的匹配情况
        accuracy = np.mean(y_pred == y_test)
        return accuracy
    #不考虑偏差太大的点
    def _remove_outliers(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
            threshold = np.percentile(densities, self.kde_threshold)  # 保留密度最高的 (100-self.kde_threshold)% 的点
            mask = densities > threshold

            X_filtered.extend(X_label[mask])
            y_filtered.extend(y[y == label][mask])

        return np.array(X_filtered), np.array(y_filtered)

    # 得到参数w和b
    def get_params(self) -> Tuple[Optional[np.ndarray], float]:
        w_value = np.linalg.norm(self.w)
        return self.w / w_value, self.b/w_value
    #选择第二个alpha
    def _select_alpha_pair(self, i: int) -> Tuple[int, bool]:
        """
        选择第二个alpha值，使用最大化步长的方式
        返回: (j, 是否在最大迭代次数内找到)
        """
        n_samples = self.X.shape[0]
        max_iter = 20
        counter = 0
        j = i
        # 初始化错误缓存
        if self.error_cache is None:
            self.error_cache = np.zeros(n_samples)
            for k in range(n_samples):
                self.error_cache[k] = self._calc_error(k)

        # 获取第i个样本的误差
        E_i = self.error_cache[i]
        # 首先尝试在非边界alpha中寻找最大步长的j
        non_bound_idx = np.where((self.alpha > 0) & (self.alpha < self.C))[0]
        max_step = 0
        if len(non_bound_idx) > 0:
            for k in non_bound_idx:
                if k == i:
                    continue
                E_k = self.error_cache[k]
                step = abs(E_i - E_k)
                if step > max_step:
                    max_step = step
                    j = k
            if j != i:
                return j, True
        # 如果在非边界中没找到合适的j，则随机选择
        while j == i and counter < max_iter:
            j = np.random.randint(0, n_samples)
            counter += 1
        return j, counter < max_iter

    def _calc_error(self, k: int) -> float:
        """
        计算第k个样本的误差
        """
        return self._decision_function_single(self.X[k]) - self.y[k]