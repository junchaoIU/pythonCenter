#coding:gbk,
#  缺失值处理

import pandas as pd  # 导入pandas库
import numpy as np  # 导入numpy库
from sklearn.impute import SimpleImputer
#Imputer方法：缺失值计算

print('生成缺失数据')
df = pd.DataFrame(np.random.randn(6, 4),
                  columns=['col1', 'col2', 'col3', 'col4'])  # 生成一份数据

df.iloc[1:3, 1] = np.nan  # 增加缺失值
df.iloc[4, 3] = np.nan  # 增加缺失值
print(df)

print('\n看哪些值缺失')
nan_all = df.isnull()  # 获得所有数据框中的N值
print(nan_all)  # 打印输出

print('\n查看哪些列缺失')
nan_col1 = df.isnull().any()  # 获得含有NA的列
nan_col2 = df.isnull().all()  # 获得全部为NA的列
print(nan_col1)  # 打印输出
print(nan_col2)  # 打印输出

print('\n1丢弃缺失值')
df2 = df.dropna()  # 直接丢弃含有NA的行记录
#df2 = df.dropna(axis = 1) #删除列
print(df2)  # 打印输出

print('\n2使用sklearn将缺失值替换为特定值')
imp_mean=SimpleImputer(strategy='constant',fill_value=111)
#mean:平均值，median：中位数，most_frequent：众数
imp_mean.fit(df)
print(imp_mean.transform(df))

print('\n3使用pandas将缺失值替换为特定值')
nan_result_pd1 = df.fillna(method='backfill')  # 用后面的值替换缺失值
nan_result_pd2 = df.fillna(method='bfill', limit=1)  # 用后面的值替代缺失值,限制每列只能替代一个缺失值
nan_result_pd3 = df.fillna(method='pad')  # 用前面的值替换缺失值
nan_result_pd4 = df.fillna(0)  # 用0替换缺失值
nan_result_pd5 = df.fillna({'col2': 1.1, 'col4': 1.2})  # 用不同值替换不同列的缺失值
nan_result_pd6 = df.fillna(df.mean()['col2':'col4'])  # 用平均数代替,选择各自列的均值替换缺失值
#打印输出
print(nan_result_pd1)  # 打印输出
print(nan_result_pd2)  # 打印输出
print(nan_result_pd3)  # 打印输出
print(nan_result_pd4)  # 打印输出
print(nan_result_pd5)  # 打印输出
print(nan_result_pd6)  # 打印输出



















