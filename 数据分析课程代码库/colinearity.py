#!/usr/bin/env python
# coding:gbk,
# 解决运营数据的共线性问题

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV, LinearRegression
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pyplot as plt

# 读取数据
data = np.loadtxt('data5.txt', delimiter='\t')  # 读取数据文件
print(data.shape)
x, y = data[:, :-1], data[:, -1]  # 切分自变量和预测变量
correlation_matrix = np.corrcoef(x, rowvar=0)  # 相关性分析
print("相关系数矩阵：\n %s" % correlation_matrix.round(2))  # 打印输出相关性结果

print('使用岭回归算法进行回归分析')
model_ridge = Ridge(alpha=1.0)  # 建立岭回归模型对象
model_ridge.fit(x, y)  # 输入x/y训练模型
print(model_ridge.coef_)  # 打印输出自变量的系数
print(model_ridge.intercept_)  # 打印输出截距
print(model_ridge.score(x, y))  # R方

print('使用普通的线性回归')
m1 = LinearRegression()
m1.fit(x, y)
print(m1.score(x, y))

print('使用rigdeCV获取最佳alpha')
model = RidgeCV(alphas=[0.1, 1.0, 10.0], store_cv_values=True)  # 通过RidgeCV可以设置多个参数值，算法使用交叉验证获取最佳参数值
model.fit(x, y)
print("Best alpha using built-in RidgeCV: %s" % model.alpha_)

model_ridge = Ridge(alpha=0.1)  # 建立岭回归模型对象
model_ridge.fit(x, y)  # 输入x/y训练模型
print(model_ridge.score(x, y))  # R方

print('\n 使用主成分回归进行回归分析')
model_pca = PCA()  # 建立PCA模型对象
data_pca = model_pca.fit_transform(x)  # 将x进行主成分分析
print(x[:2, :])
k = model_pca.inverse_transform(data_pca) # 降维后的数据转化为原始数据
print(k[:2, :])
ratio_cumsm = np.cumsum(
    model_pca.explained_variance_ratio_)  # 得到所有主成分方差占比的累积数据
print(ratio_cumsm)  # 打印输出所有主成分方差占比累积
rule_index = np.where(ratio_cumsm > 0.8)  # 获取方差占比超过0.8的所有索引值
print("rule_index:{}".format(rule_index))
min_index = rule_index[0][0]  # 获取最小索引值
print("min_index:{}".format(min_index))
data_pca_result = data_pca[:, :min_index + 1]  # 根据最小索引值提取主成分
model_liner = LinearRegression()  # 建立回归模型对象
model_liner.fit(data_pca_result, y)  # 输入主成分数据和预测变量y并训练模型
print("自变量的系数:{}".format(model_liner.coef_))  # 打印输出自变量的系数
print(model_liner.intercept_)  # 打印输出截距
print(model_liner.score(data_pca_result, y))



