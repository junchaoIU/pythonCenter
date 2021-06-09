# -*- coding: utf-8 -*-

# 导入库
import copy  # 复制库

import numpy as np  # numpy库
import pandas as pd  # pandas库
from imblearn.over_sampling import SMOTE  # 过抽样处理库SMOTE
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier, GradientBoostingClassifier, \
    RandomForestClassifier,  BaggingClassifier  # 四种集成分类库和投票方法库
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score  # 导入交叉检验算法
from sklearn.preprocessing import LabelEncoder  # 字符串转数值
from sklearn.metrics import accuracy_score,confusion_matrix

# 数据审查和预处理函数
# 基本状态查看
def set_summary(df):
    '''
    查看数据集后2条数据、数据类型、描述性统计
    :param df: 数据框
    :return: 无
    '''
    print('{:*^60}'.format('Data overview:'))
    print(df.tail(2).T)  # 打印原始数据后2条
    print('{:*^60}'.format('Data dtypes:'))
    print(df.dtypes)  # 打印数据类型
    print('{:*^60}'.format('Data DESC:'))
    print(df.describe().round(2).T)  # 打印原始数据基本描述性信息


# 缺失值审查
def na_summary(df):
    '''
    查看数据集的缺失数据列、行记录数
    :param df: 数据框
    :return: 无
    '''
    na_cols = df.isnull().any(axis=0)  # 查看每一列是否具有缺失值
    print('{:*^60}'.format('NA Cols:'))
    print(na_cols)  # 查看具有缺失值的列
    print('Total number of NA lines is: {0}'.format(df.isnull().any(axis=1).sum()))  # 查看具有缺失值的行总记录数


# 类样本均衡审查
def label_samples_summary(df):
    '''
    查看每个类的样本量分布
    :param df: 数据框
    :return: 无
    '''
    print('{:*^60}'.format('Labesl samples count:'))
    print(df.iloc[:, 0].groupby(df.iloc[:, -1]).count())


# 字符串分类转数值分类
def label_encoder(data, model_list=None, train=True):
    '''
    将特征中的字符串分类转换为数值分类
    :param data: 输入数据集
    :param model_list: LabelEncoder对象列表，在训练阶段产生
    :param train: 是否为训练阶段
    :return: 训练阶段产生训练后的LabelEncoder对象列表和转换后的数据，预测阶段产生转换后的数据
    '''
    convert_cols = ['cat', 'attribution', 'pro_id', 'pro_brand', 'order_source', 'pay_type',
                    'use_id','city']  # 定义要转换的列
    value_list = []  # 存放转换后的数据
    if train:
        model_list = []  # 存放每个特征转换的实例对象
        model_label_encoder = LabelEncoder()
        for i in convert_cols:
            model_label_encoder.fit(data[i])
            value_list.append(model_label_encoder.transform(data[i]))
            model_list.append(copy.copy(model_label_encoder))
        convert_matrix = np.array(value_list).T
        return model_list, convert_matrix
    else:
        for ind, j in enumerate(convert_cols):
            value_list.append(model_list[ind].transform(data[j]))
        convert_matrix = np.array(value_list).T
        return convert_matrix 
        

# 时间属性拓展
def datetime2int(data):
    '''
    将日期和时间数据拓展出其他属性，例如星期几、周几、小时、分钟等。
    :param data: 数据集
    :return: 拓展后的属性矩阵
    '''
    date_set = [pd.datetime.strptime(dates, '%Y-%m-%d') for dates in
                data['order_date']]  # 将data中的order_date列转换为特定日期格式
    '''date_set = [pd.to_datetime(dates, format='%Y-%m-%d') for dates in
                data['order_date']] '''
    
    weekday_data = [data.weekday() for data in date_set]  # 周几
    daysinmonth_data = [data.day for data in date_set]  # 当月几号
    month_data = [data.month for data in date_set]  # 月份

    time_set = [pd.datetime.strptime(times, '%H:%M:%S') for times in
                data['order_time']]  # 将data中的order_time列转换为特定时间格式
    second_data = [data.second for data in time_set]  # 秒
    minute_data = [data.minute for data in time_set]  # 分钟
    hour_data = [data.hour for data in time_set]  # 小时

    final_set = [weekday_data, daysinmonth_data, month_data, second_data, minute_data,
                 hour_data]  # 将属性列表批量组合
    final_matrix = np.array(final_set).T  # 转换为数组并转置
    return final_matrix


# 样本均衡
def sample_balance(X, y):
    '''
    使用SMOTE方法对不均衡样本做过抽样处理
    :param X: 输入特征变量X
    :param y: 目标变量y
    :return: 均衡后的X和y
    '''
    model_smote = SMOTE()  # 建立SMOTE模型对象
    x_smote_resampled, y_smote_resampled = model_smote.fit_resample(X, y)  # 输入数据并作过抽样处理
    return x_smote_resampled, y_smote_resampled


# 数据应用
# 定义特殊字段数据格式
dtypes = {'order_id': np.object,
          'pro_id': np.object,
          'use_id': np.object}
raw_data = pd.read_table('./data/abnormal_orders.txt', delimiter=',', dtype=dtypes)  # 读取数据集

# 数据审查
set_summary(raw_data)  # 基本状态查看
na_summary(raw_data)  # 缺失值审查
label_samples_summary(raw_data)  # 类样本分布审查

# 数据预处理
drop_na_set = raw_data.dropna()  # 丢弃带有NA值的数据行
X_raw, y_raw = drop_na_set.iloc[:, 1:-1], drop_na_set.iloc[:, -1]  # 分割输入变量X和y
model_list, convert_matrix = label_encoder(X_raw)  # 字符串分类转整数型分类
#print(convert_matrix)
datetime2int_data = datetime2int(X_raw)  # 拓展日期时间属性
#print(datetime2int_data)
combine_set = np.hstack((convert_matrix, datetime2int_data))  # 合并转换后的分类和拓展后的日期数据集
constant_set = X_raw[['total_money', 'total_quantity']]  # 原始连续数据变量
X_combine = np.hstack((combine_set, constant_set))  # 再次合并数据集
# 相关性分析
X_combine=pd.DataFrame(X_combine)
print('{:*^60}'.format('Correlation Analyze:'))
print(X_combine.corr().round(2))  # 输出所有输入特征变量以及预测变量的相关性矩阵

X, y = sample_balance(X_combine, y_raw)  # 样本均衡处理
'''X=X_combine
y=y_raw'''

# 组合分类模型交叉检验
model_rf = RandomForestClassifier(n_estimators=20, random_state=0)  # 随机森林分类模型对象
model_lr = LogisticRegression(max_iter=1000,random_state=0)  # 逻辑回归模型对象
#model_BagC = BaggingClassifier(n_estimators=20,random_state=0)  # Bagging分类模型对象
#model_gdbc = GradientBoostingClassifier(max_features=0.8, random_state=0)  # GradientBoosting分类模型对象
estimators = [('randomforest', model_rf), ('Logistic', model_lr)]  # 建立组合评估器列表
model_vot = VotingClassifier(estimators=estimators, voting='soft', weights=[0.5, 0.5],
                             n_jobs=-1)  # 建立组合评估模型
cv = StratifiedKFold(2)  # 设置交叉检验方法
cv_score = cross_val_score(model_vot, X, y, cv=cv)  # 交叉检验
print('{:*^60}'.format('Cross val scores:'))
print(cv_score)  # 打印每次交叉检验得分
print('Mean scores is: %.2f' % cv_score.mean())  # 打印平均交叉检验得分
model_vot.fit(X, y)  # 模型训练
y_predicttrain = model_vot.predict(X)  # 预测结果
print('混淆矩阵：')
print(confusion_matrix(y,y_predicttrain))

# 新数据集做预测
X_raw_data = pd.read_csv('./data/new_abnormal_orders.csv', dtype=dtypes)  # 读取要预测的数据集
X_raw_new = X_raw_data.iloc[:, 1:]  # 分割输入变量X，并丢弃订单ID列
convert_matrix_new = label_encoder(X_raw_new, model_list, False)  # 字符串分类转整数型分类
datetime2int_data_new = datetime2int(X_raw_new)  # 日期时间转换
combine_set_new = np.hstack((convert_matrix_new, datetime2int_data_new))  # 合并转换后的分类和拓展后的日期数据集
constant_set_new = X_raw_new[['total_money', 'total_quantity']]  # 原始连续数据变量
X_combine_new = np.hstack((combine_set_new, constant_set_new))  # 再次合并数据集
y_predict = model_vot.predict(X_combine_new)  # 预测结果
print('{:*^60}'.format('Predicted Labesls:'))
print(y_predict)  # 打印预测值
