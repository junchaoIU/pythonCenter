# -------------------------------------------------------------------------------
# Description:  
# Reference:
# Name:   work3
# Author: wujunchao
# Date:   2021/6/2
# -------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from IPython.display import display
import mglearn

# 去除异常值，离群点
def abnormal_detect(df):
    # 通过Z-Score方法判断异常值
    df_zscore = df.copy()  # 复制一个用来存储Z-score得分的数据框
    cols = df.columns  # 获得数据框的列名

    for col in cols:  # 循环读取每列
        df_col = df[col]  # 得到每列的值
        z_score = (df_col - df_col.mean()) / df_col.std()  # 计算每列的Z-score得分（（每个元素-该列平均值）/该列标准差）
        df_zscore[col] = z_score.abs() > 2.2  # 判断Z-score得分的绝对值是否大于2.2，如果是则是True，否则为False
    # print(df_zscore)  # 打印输出

    # 删除异常值所在的行
    for col in cols:  # 循环读取每列删除
        df_drop = df[df_zscore[col] == False]
    return df_drop

# 文字字符型转数字型
def str_num(data_col):
    data_col = data_col.values
    data_col_list = list(set(data_col))
    for i in range(0,len(data_col)):
        data_col[i] = data_col_list.index(data_col[i])
    data_col = pd.DataFrame(data_col)
    print(data_col.head(5))
    return data_col

# 读取并预处理数据
def read_data(data_path):
    print('=' * 20 + "数据预处理" + '=' * 20)
    # 打开文件
    data = pd.read_csv(data_path)
    # 查看训练集和测试集前5行
    print("训练集和测试集前5行:")
    print(data.head(5))
    # 查看训练集和测试集样本量及特征量
    print('dataset: {}'.format(data.shape))
    # 查看数据中缺失值的量
    print("查看数据中缺失值的量：")
    print(data.isnull().sum())
    # 删除有空值的行
    data = data.dropna()

    # 删除无用特征
    data.drop(['order_id',"pro_id","use_id"], axis=1, inplace=True)
    print(data)

    # 查看字符串类别特征的类型数量
    data["cat"] = str_num(data["cat"])
    data["attribution"] = str_num(data["attribution"])
    data["pro_brand"] = str_num(data["pro_brand"])
    data["pro_brand"] = str_num(data["order_source"])
    data["pro_brand"] = str_num(data["pay_type"])
    data["pro_brand"] = str_num(data["city"])

    # Z-Score方法判断异常值并去除
    # data = abnormal_detect(data)

    # 日期格式转换
    data['order_date'] = pd.to_datetime(data['order_date'],format='%Y-%m-%d')  # 将字符串转换为日期格式
    print('data Dtypes:')
    print(data.dtypes)  # 打印输出数据框所有列的数据类型
    print('-' * 60)

    print(data.head(5))
    return data

train = read_data("abnormal_orders.txt")
X_train = train.drop(['abnormal_label'], axis=1)
y_train = train["abnormal_label"]

X_test = read_data("new_abnormal_orders.csv")

print(X_train.head(5))
print(X_test.head(5))