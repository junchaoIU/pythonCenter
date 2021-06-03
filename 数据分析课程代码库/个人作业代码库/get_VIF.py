# -*-coding:utf-8-*-
# @Time    : 2021/04/08 4:18
# @Author  : Wu Junchao
# @Software: PyCharm
# pip --default-timeout=100 install 库名称 -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com
"""
请用python写一个求容忍度/膨胀因子的程序，具体功能写成函数，并在程序中调用
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# 测试数据X
X = pd.DataFrame(
    {'a': [1, 1, 2, 3, 4],
     'b': [2, 2, 3, 2, 1],
     'c': [4, 6, 7, 8, 9],
     'd': [4, 3, 4, 5, 4]}
)

"""
计算vif的函数，输入X为DataFrame二维数组
VIF = 1/(1-R^2)
"""
def vif(X):
    for i in X.columns:
        y = X[i].values # 自变量i
        x = X.drop(i,axis=1).values  # i之外其余自变量
        lr = LinearRegression().fit(x, y)   # LinearRegression，模型构建，训练
        R2 = lr.score(x, y)   # R^2
        r = 1.0 - float(R2) # r:容忍度
        # 排除R2等于1时的问题
        if(R2 == 1.0):
            r = "无限接近于0"
            vif = "无穷大"
            text = "存在严重多重共线性"
        else:
            vif = 1.0 / r  # n:方差膨胀系数
            # 判断是否存在共线性，一般以10为界定，部分学者也以5界定
            if vif>10:
                text = "存在严重多重共线性"
            else:
                text = "无异常"

        print("{}的容忍度为{},方差膨胀系数为:{},{}\n".format(i,r,vif,text))

vif(X)
