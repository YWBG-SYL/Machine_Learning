#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@IDE     ：PyCharm 
@Author  ：原味不改
@Date    ：2024/10/27 13:37 
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

matplotlib.use('TkAgg')
from pylab import mpl

# 设置中文显示字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False

#设置最大序号和时间间隔T
max=4001
T=17
#初始化x序列
x=np.zeros(max+T)


#定义离散的混沌时间序列函数
def mackey_glass(x,T,a,b,h,n,max):
    for t in range(T,max+T-1):
        x[t + 1] = x[t] + h * (-b * x[t] + a * x[t - T] / (1 + x[t - T] ** n))
    return x
#使用常数初始化
def static_init(x,num):
    x[:T]=num
    return x

#使用正弦初始化
def sin_init(x):
    index_t = np.linspace(0, 2 * np.pi, T)
    x[:T] = np.sin(index_t)
    return x

#使用余弦初始化
def cos_init(x):
    index_t= np.linspace(0, 2 * np.pi, T)
    x[:T]=np.cos(index_t)
    return x


def main():
    #初始化参数
    a=0.2
    b=0.1
    n=10
    h=1
    #初始化T之前的序列
    res=static_init(x,1)
    res1=mackey_glass(res,T,a,b,h,n,max)
    #绘图
    t_step = np.arange(0, max+T)
    plt.figure(figsize=(9, 6))
    plt.scatter(t_step, res1, color='blue', marker='o', s=10, edgecolors='w')
    # 添加标签和标题
    plt.xlabel('t')
    plt.ylabel('value')
    plt.title('时间混沌序列')

    # 添加网格
    plt.grid()

    # 显示图形
    plt.show()
    #保存到csv文件中
    df = pd.DataFrame({'Time': t_step[1:max], 'Value': res1[T+1:max+T]})
    # 保存为 CSV 文件
    df.to_csv('./res/chaotic_time_series.csv', index=False)
    print("生成的混沌时间序列已保存")

if __name__=='__main__':
    main()
