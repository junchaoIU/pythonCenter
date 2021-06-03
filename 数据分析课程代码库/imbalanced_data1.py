#!/usr/bin/env python
#coding:gbk,
import pandas as pd
from imblearn.over_sampling import SMOTE  # 过抽样处理库SMOTE
from imblearn.under_sampling import RandomUnderSampler  # 欠抽样处理库RandomUnderSampler
from sklearn.svm import SVC  # SVM中的分类算法SVC

# 导入数据文件
df = pd.read_table('data22.txt', sep='\t',
                   names=['col1', 'col2', 'col3', 'col4', 'col5',
                          'label'])  # 读取数据文件
x = df.iloc[:, :-1]  # 切片，得到输入x
y = df.iloc[:, -1]  # 切片，得到标签y
print(df.iloc[0:5,:])
groupby_data_orgianl = df.groupby('label').count()  # 对label做分类汇总
print('原始数据集样本分类分布')
print(groupby_data_orgianl)  

# 使用SMOTE方法进行过抽样处理
model_smote = SMOTE()  # 建立SMOTE模型对象
x_smote_resampled, y_smote_resampled = model_smote.fit_sample(x,y)  # 输入数据并作过抽样处理
x_smote_resampled = pd.DataFrame(x_smote_resampled,
                                 columns=['col1', 'col2', 'col3', 'col4',
                                          'col5'])  # 将数据转换为数据框并命名列名
y_smote_resampled = pd.DataFrame(y_smote_resampled,
                                 columns=['label'])  # 将数据转换为数据框并命名列名
smote_resampled = pd.concat([x_smote_resampled, y_smote_resampled],
                            axis=1)  # 按列合并数据框
groupby_data_smote = smote_resampled.groupby('label').count()  # 对label做分类汇总
print(groupby_data_smote)  # 打印输出经过SMOTE处理后的数据集样本分类分布

# 使用RandomUnderSampler方法进行欠抽样处理
model_RandomUnderSampler = RandomUnderSampler()  # 建立RandomUnderSampler模型对象
x_RandomUnderSampler_resampled, y_RandomUnderSampler_resampled = model_RandomUnderSampler.fit_sample(
x,y)  # 输入数据并作欠抽样处理
x_RandomUnderSampler_resampled = pd.DataFrame(x_RandomUnderSampler_resampled,
                                              columns=['col1', 'col2', 'col3',
                                                       'col4',
                                                       'col5'])  # 将数据转换为数据框并命名列名
y_RandomUnderSampler_resampled = pd.DataFrame(y_RandomUnderSampler_resampled,
                                              columns=['label'])  # 将数据转换为数据框并命名列名
RandomUnderSampler_resampled = pd.concat([x_RandomUnderSampler_resampled, 
y_RandomUnderSampler_resampled],axis=1)  # 按列合并数据框
groupby_data_RandomUnderSampler = RandomUnderSampler_resampled.groupby(
    'label').count()  # 对label做分类汇总
print(groupby_data_RandomUnderSampler)  # 打印输出经过RandomUnderSampler处理后的数据集样本分类分布

# 使用SVM的权重调节处理不均衡样本
model_svm = SVC(class_weight='balanced')  # 创建SVC模型对象并指定类别权重
model_svm.fit(x, y)  # 输入x和y并训练模型
print(y[0:5])
ynew=model_svm.predict(x)
print(ynew[0:5])
