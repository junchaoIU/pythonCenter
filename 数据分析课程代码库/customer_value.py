#coding:gbk,

# 导入库
import time  # 导入时间库
from datetime import datetime

import numpy as np  # 导入numpy库
import pandas as pd  # 导入pandas库

# 读取数据
dtypes = {'ORDERDATE':object, 'ORDERID': object, 'AMOUNTINFO': np.float32}  # 设置每列数据类型,字典
raw_data = pd.read_csv('sales.csv', dtype=dtypes,index_col='USERID')  # 读取数据文件
print(raw_data.dtypes) # 字段类型
print('-' * 60)

# 数据审查和校验
# 数据概览
print('Data Overview:')
print(raw_data.head(4))  # 打印原始数据前4条
raw_data.tail()  # # 打印原始数据吼后5条
print('-' * 30)
print('Data DESC:')
print(raw_data.describe())  # 打印原始数据基本描述性信息
print('-' * 60)

# 缺失值审查
na_cols = raw_data.isnull().any(axis=0)  # 查看每一列是否具有缺失值
print('NA Cols:')
print(na_cols)  # 查看具有缺失值的列
print('-' * 30)
na_lines = raw_data.isnull().any(axis=1)  # 查看每一行是否具有缺失值
print('NA Recors:')
print('Total number of NA lines is: {0}'.format(na_lines.sum()))  # 查看具有缺失值的行总记录数
print(raw_data[na_lines])  # 只查看具有缺失值的行信息
print('-' * 60)

# 数据异常、格式转换和处理
# 异常值处理
sales_data = raw_data.dropna()  # 丢弃带有缺失值的行记录
sales_data = sales_data[sales_data['AMOUNTINFO'] > 1]  # 丢弃订单金额<=1的记录

# 日期格式转换
sales_data['ORDERDATE'] = pd.to_datetime(sales_data['ORDERDATE'], format='%Y-%m-%d')  # 将字符串转换为日期格式
print('Sales_data Dtypes:')
print(sales_data.dtypes)  # 打印输出数据框所有列的数据类型
print('-' * 60)

# 数据转换
recency_value = sales_data['ORDERDATE'].groupby(sales_data.index).max()  # 计算原始最近一次订单时间
frequency_value = sales_data['ORDERID'].groupby(sales_data.index).count()  # 计算原始订单频率
monetary_value = sales_data['AMOUNTINFO'].groupby(sales_data.index).sum()  # 计算原始订单总金额


# 计算RFM得分
# 分别计算R、F、M得分
deadline_date = datetime(2017, 1, 1)  # 指定一个时间节点，用于计算其他时间与该时间的距离
r_interval = (deadline_date - recency_value).dt.days  # 计算R间隔
# 分箱
r_score = pd.cut(r_interval, 5, labels=[5, 4, 3, 2, 1])  # 计算R得分
f_score = pd.cut(frequency_value, 5, labels=[1, 2, 3, 4, 5])  # 计算F得分
m_score = pd.cut(monetary_value, 5, labels=[1, 2, 3, 4, 5])  # 计算M得分
print('-' * 60)

# R、F、M数据合并
rfm_list = [r_score, f_score, m_score]  # 将r、f、m三个维度组成列表
rfm_cols = ['r_score', 'f_score', 'm_score']  # 设置r、f、m三个维度列名
rfm_pd = pd.DataFrame(np.array(rfm_list).transpose(), dtype=np.int32, columns=rfm_cols,
                      index=frequency_value.index)  # 建立r、f、m数据框
print('RFM Score Overview:')
print(rfm_pd.head(4))
print('-' * 60)

# 计算RFM总得分
# 方法一：加权得分
rfm_pd['rfm_wscore'] = rfm_pd['r_score'] * 0.6 + rfm_pd['f_score'] * 0.3 + rfm_pd['m_score'] * 0.1
# 方法二：RFM组合
rfm_pd_tmp = rfm_pd.copy()
rfm_pd_tmp['r_score'] = rfm_pd_tmp['r_score'].astype(np.str)
rfm_pd_tmp['f_score'] = rfm_pd_tmp['f_score'].astype(np.str)
rfm_pd_tmp['m_score'] = rfm_pd_tmp['m_score'].astype(np.str)
# 拼接
rfm_pd['rfm_comb'] = rfm_pd_tmp['r_score'].str.cat(rfm_pd_tmp['f_score']).str.cat(
    rfm_pd_tmp['m_score'])

# 打印输出和保存结果
# 打印结果
print('Final RFM Scores Overview:')
print(rfm_pd.head(4))  # 打印数据前4项结果
print('-' * 30)
print('Final RFM Scores DESC:')
print(rfm_pd.describe())

# 保存RFM得分到本地文件
rfm_pd.to_csv('sales_rfm_score.csv')  # 保存数据为csv

import matplotlib.pyplot as plt

print(sales_data)
print(sales_data.dtypes)
sales_data['month'] = sales_data['ORDERDATE'].dt.month
sales_month = sales_data['AMOUNTINFO '].groupby(sales_data['month']).sum()

# plt.title("2016", fontsize=20)
# squares=[1, 4, 9, 16, 25,45,15,589,26,36,66,88]
# x=[1, 2, 3, 4, 5,6,7,8,9,10,11,12]
# plt.plot(x, squares)
# plt.show()
