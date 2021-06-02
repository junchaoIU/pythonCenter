#!/usr/bin/env python
#coding:gbk,
#决策树
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score,auc,confusion_matrix,f1_score,precision_score,recall_score,roc_curve#分类指标库
import prettytable
import matplotlib.pyplot as plt

raw_data=np.loadtxt('./data/classification.csv', delimiter=',',skiprows=1)

X=raw_data[:,:-1]
print(type(X))
y=raw_data[:,-1]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3,random_state=0)#将数据集分为训练集和测试集

#训练分类模型
model_tree=tree.DecisionTreeClassifier(random_state=0)#建立决策树模型
model_tree.fit(X_train,y_train)
pre_y=model_tree.predict(X_test)

n_samples,n_features=X.shape
print('samples: %d \t features: %d' % (n_samples,n_features))
print(70 * '-')

#混淆矩阵
confusion_m=confusion_matrix(y_test,pre_y)
confusion_matrix_table=prettytable.PrettyTable()
confusion_matrix_table.add_row(confusion_m[0,:])
confusion_matrix_table.add_row(confusion_m[1,:])
print('confusion matrix')
print(confusion_matrix_table)

# 核心评估指标
y_score = model_tree.predict_proba(X_test)  # 获得决策树的预测概率
fpr, tpr, thresholds = roc_curve(y_test, y_score[:, 1])  # ROC
auc_s = auc(fpr, tpr)  # AUC
accuracy_s = accuracy_score(y_test, pre_y)  # 准确率
precision_s = precision_score(y_test, pre_y)  # 精确度
recall_s = recall_score(y_test, pre_y)  # 召回率
f1_s = f1_score(y_test, pre_y)  # F1得分
core_metrics = prettytable.PrettyTable()  # 创建表格实例
core_metrics.field_names = ['auc', 'accuracy', 'precision', 'recall', 'f1']  # 定义表格列名
core_metrics.add_row([auc_s, accuracy_s, precision_s, recall_s, f1_s])  # 增加数据
print('core metrics')
print(core_metrics)  # 打印输出核心评估指标

# 模型效果可视化
names_list = ['age', 'gender', 'income', 'rfm_score']  # 分类模型维度列表
color_list = ['r', 'c', 'b', 'g']  # 颜色列表
plt.figure()  # 创建画布
# 子网格1：ROC曲线
plt.subplot(1, 2, 1)  # 第一个子网格
plt.plot(fpr, tpr, label='ROC')  # 画出ROC曲线
plt.plot([0, 1], [0, 1], linestyle='--', color='k', label='random chance')  # 画出随机状态下的准确率线
plt.title('ROC')  # 子网格标题
plt.xlabel('false positive rate')  # X轴标题
plt.ylabel('true positive rate')  # y轴标题
plt.legend(loc=0)
# 子网格2：指标重要性
feature_importance = model_tree.feature_importances_  # 获得指标重要性
print(feature_importance)
plt.subplot(1, 2, 2)  # 第二个子网格
plt.bar(np.arange(feature_importance.shape[0]), feature_importance, tick_label=names_list, color=color_list)  # 画出条形图

plt.title('feature importance')  # 子网格标题
plt.xlabel('features')  # x轴标题
plt.ylabel('importance')  # y轴标题
plt.suptitle('classification result')  # 图形总标题
plt.show()  # 展示图形


# 模型应用
X_new = [[40, 0, 55616, 0], [17, 0, 55568, 0], [55, 1, 55932, 1]]
print('classification prediction')
for i, data in enumerate(X_new):
    y_pre_new = model_tree.predict(np.array(data).reshape(1, -1))
    print('classification for %d record is: %d' % (i + 1, y_pre_new))

