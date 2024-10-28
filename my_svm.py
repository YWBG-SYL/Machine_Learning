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
    def __init__(self, max_iter=100, kernel='linear', C=1.0, tol=1e-3,gamma=1.0,random_seed: int = None):
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
    #选择alpha乘子
    # def _select_alpha_pair(self, i: int) -> Tuple[int, bool]:
    #     n_samples = self.X.shape[0]
    #     j = i
    #     max_iter = 20
    #     counter = 0
    #     while j == i and counter < max_iter:
    #         j = np.random.randint(0, n_samples)
    #         counter += 1
    #     return j, counter < max_iter
    def _select_alpha_pair(self, i: int) -> Tuple[int, bool]:
        n_samples = self.X.shape[0]
        # 生成一个不包括 i 的随机索引 j
        possible_indices = list(range(n_samples))
        possible_indices.remove(i)  # 移除当前索引 i
        j = np.random.choice(possible_indices)  # 从剩余的索引中随机选择
        return j, True  # 始终返回 True，因为我们已经成功选择了 j

    #计算单个样本的决策值
    def _decision_function_single(self, x: np.ndarray) -> float:
        result = 0.0
        for i in range(len(self.alpha)):
            if self.alpha[i] > 0:  # 只考虑支持向量
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

    #alpha更新函数
    # def _update_alpha_pair(self, i: int, j: int, K: np.ndarray) -> bool:
    #     if i == j:
    #         return False
    #
    #     alpha_i_old = self.alpha[i].copy()
    #     alpha_j_old = self.alpha[j].copy()
    #     y_i, y_j = self.y[i], self.y[j]
    #
    #     # 计算上下界
    #     if y_i != y_j:
    #         L = max(0, alpha_j_old - alpha_i_old)
    #         H = min(self.C, self.C + alpha_j_old - alpha_i_old)
    #     else:
    #         L = max(0, alpha_i_old + alpha_j_old - self.C)
    #         H = min(self.C, alpha_i_old + alpha_j_old)
    #
    #     if L == H:
    #         return False
    #
    #     # 计算eta
    #     eta = 2 * K[i, j] - K[i, i] - K[j, j]
    #     if eta >= 0:
    #         return False
    #
    #     # 计算 E_i 和 E_j
    #     E_i = self._decision_function_single(self.X[i]) - y_i
    #     E_j = self._decision_function_single(self.X[j]) - y_j
    #
    #     # 更新alpha_j
    #     self.alpha[j] = alpha_j_old - y_j * (E_i - E_j) / eta
    #
    #     # 修剪alpha_j
    #     self.alpha[j] = np.clip(self.alpha[j], L, H)
    #
    #     if abs(self.alpha[j] - alpha_j_old) < self.tol:
    #         return False
    #
    #     # 更新alpha_i
    #     self.alpha[i] = alpha_i_old + y_i * y_j * (alpha_j_old - self.alpha[j])
    #
    #     # 更新b
    #     b1 = self.b - E_i - y_i * (self.alpha[i] - alpha_i_old) * K[i, i] - \
    #          y_j * (self.alpha[j] - alpha_j_old) * K[i, j]
    #     b2 = self.b - E_j - y_i * (self.alpha[i] - alpha_i_old) * K[i, j] - \
    #          y_j * (self.alpha[j] - alpha_j_old) * K[j, j]
    #
    #     if 0 < self.alpha[i] < self.C:
    #         self.b = b1
    #     elif 0 < self.alpha[j] < self.C:
    #         self.b = b2
    #     else:
    #         self.b = (b1 + b2) / 2
    #
    #     return True
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
        if eta >= 0:
            return False

        # 计算 E_i 和 E_j，并缓存
        E_i = self._decision_function_single(self.X[i]) - y_i
        E_j = self._decision_function_single(self.X[j]) - y_j

        # 更新 alpha_j
        self.alpha[j] = np.clip(alpha_j_old - y_j * (E_i - E_j) / eta, L + self.tol, H - self.tol)

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

        if 0 < self.alpha[i] < self.C:
            self.b = b1
        elif 0 < self.alpha[j] < self.C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2

        # 更新缓存
        # self.error_cache[i] = self._decision_function_single(self.X[i]) - y_i
        # self.error_cache[j] = self._decision_function_single(self.X[j]) - y_j

        return True
    #模型训练
    # def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVM':
    #     self.X = X
    #     self.y = y
    #     n_samples = X.shape[0]
    #     self.alpha = np.zeros(n_samples)
    #
    #     # 计算核矩阵
    #     K = self._get_kernel_matrix()
    #
    #     # SMO算法主循环
    #     for _ in range(self.max_iter):
    #         alpha_changed = 0
    #
    #         for i in range(n_samples):
    #             # 选择第二个alpha
    #             j, success = self._select_alpha_pair(i)
    #             if not success:
    #                 continue
    #
    #             # 更新选中的alpha对
    #             if self._update_alpha_pair(i, j, K):
    #                 alpha_changed += 1
    #
    #         if alpha_changed == 0:
    #             break
    #
    #     # 训练完成后计算权重向量（仅对线性核）
    #     if self.kernel == 'linear':
    #         self._calculate_w()
    #
    #     return self
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVM':
        # 输入数据检查
        assert X.shape[0] == y.shape[0], "样本数量与标签数量不一致"
        assert len(X.shape) == 2, "X 应该是一个二维数组"
        self.X = X
        self.y = y
        n_samples = X.shape[0]
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
    #得到参数w和b
    def get_params(self) -> Tuple[Optional[np.ndarray], float]:
        w_value = np.linalg.norm(self.w)
        return self.w/w_value, self.b/w_value

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


