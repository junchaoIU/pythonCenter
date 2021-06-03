#!/usr/bin/env python
#coding:gbk,
# PCA降维
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

# 读取数据文件
data = np.loadtxt('data1.txt')  # 读取文本数据文件
x = data[:, :-1]  # 获得输入的x
y = data[:, -1]  # 获得目标变量y
print(np.size(x,1))#获得输入的维度
print(np.size(x,0))
print(x[0], y[0])  # 打印输出x和y的第一条记录

print('\n使用sklearn的DecisionTreeClassifier判断变量重要性')
model_tree = DecisionTreeClassifier(random_state=0)  # 建立分类决策树模型对象
model_tree.fit(x, y)  # 将数据集的维度和目标变量输入模型
feature_importance = model_tree.feature_importances_  # 获得所有变量的重要性得分
print('The importance score of each parameter:')
print(feature_importance)  # 打印输出

print('\n使用sklearn的PCA进行维度转换')
model_pca = PCA()  # 建立PCA模型对象
#n_components=3
#n_components='mle'
model_pca.fit(x)  # 用数据集训练PCA模型
model_pca.transform(x)  #将X转换成降维后的数据
components = model_pca.components_  # 获得转换后的所有主成分
components_var = model_pca.explained_variance_  # 获得各主成分的方差
components_var_ratio = model_pca.explained_variance_ratio_  # 获得各主成分的方差占比
print(components[:2])  # 打印输出前2个主成分
print(components_var[:2])  # 打印输出前2个主成分的方差
print(components_var_ratio)  # 打印输出所有主成分的方差占比




