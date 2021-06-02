# -*- coding: utf-8 -*-

# 导入库
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.feature_selection import SelectPercentile,f_classif 


# 基本状态查看
def set_summary(df):
    '''
    查看数据集的记录数、维度数、前2条数据、描述性统计和数据类型
    :param df: 数据框
    :return: 无
    '''
    print('Data Overview')
    print('Records: {0}\tDimension:{1}'.format(df.shape[0], (df.shape[1] - 1)))  # 打印数据集X形状
    print('-' * 30)
    print(df.head(2))  # 打印前2条数据
    print('-' * 30)
    print('Data DESC')
    print(df.describe().T)  # 打印数据基本描述性信息
    '''print('Data Dtypes')
    print(df.dtypes)  # 打印数据类型
    print('-' * 60)'''


# 缺失值审查
def na_summary(df):
    '''
    查看数据集的缺失数据列、行记录数
    :param df: 数据框
    :return: 无
    '''
    na_cols = df.isnull().any(axis=0)  # 每一列是否具有缺失值
    print('NA Cols:')
    print(na_cols)  # 查看具有缺失值的列
    print('-' * 30)
    '''print('valid records for each Cols:')
    print(df.count())  # 查看每一列有效值（非NA）的记录数
    print('-' * 30)'''
    print('Total number of NA lines is: {0}'.format(df.isnull().any(axis=1).sum()))  # 查看具有缺失值的行总记录数
    print('-' * 30)


# 类样本均衡审查
def label_summary(df):
    '''
    查看每个类的样本量分布
    :param df: 数据框
    :return: 无
    '''
    print('Labels samples count:')
    print(df['value_level'].groupby(df['response']).count())  # 以response为分类汇总维度对value_level列计数统计
    print('-' * 60)


# 数据预处理
def type_con(df):
	'''转换目标列的数据为特定数据类型'''
	var_list={'edu':'int32',
	          'user_level':'int32',
	          'industry':'int32',
	          'value_level':'int32',
	          'act_level':'int32',
	          'sex':'int32',
	          'region':'int32',
	          'age':'int32'}#字典，定义要转换的列及其数据类型
	for var,type in var_list.items():
		df[var]=df[var].astype(type)
		return df
		 
# NA值替换
def na_replace(df):
    '''
    将数据集中的NA值使用自定义方法替换
    :param df: 数据框
    :return: NA值替换后的数据框
    '''
    na_rules = {'age': df['age'].mean(),
                'total_pageviews': df['total_pageviews'].mean(),
                'edu': df['edu'].median(),
                'edu_ages': df['edu_ages'].median(),
                'user_level': df['user_level'].median(),
                'industry': df['user_level'].median(),
                'act_level': df['act_level'].median(),
                'sex': df['sex'].mode()[0],
                'red_money': df['red_money'].mean(),
                'region': df['region'].median()
                }  # 字典：定义各个列数据转换方法
    df = df.fillna(na_rules)  # 使用指定方法填充缺失值
    print('Check NA exists:')
    print((df.isnull().any().sum()))  # 查找是否还有缺失值
    print(('-' * 30))
    return df

#数据二值化
def symbol_con(df,enc_object=None,train=True):
	convert_cols=['user_level','industry','value_level','act_level',
	'sex']
	df_con=df[convert_cols]
	df_org=df[['age','total_pageviews','edu_ages','blue_money','red_money',
	'work_hours']].values
	if train==True:
		enc=OneHotEncoder()
		enc.fit(df_con)
		df_con_new=enc.transform(df_con).toarray()	
		new_matrix=np.hstack((df_con_new,df_org))#将未转换的数据与转换的数据合并
		return new_matrix,enc
	else:
		df_con_new=enc_object.transform(df_con).toarray()
		new_matrix=np.hstack((df_con_new,df_org))
		return new_matrix

#数据降维
def dimen_reduce(dfx,dfy,percentile_n=50):
	transform_m=SelectPercentile(f_classif,percentile=percentile_n)
	transform_m.fit(dfx,dfy)
	return transform_m.get_support(True)	
	

# 数据应用
# 加载数据集
raw_data = pd.read_excel('./data/order1.xlsx',sheet_name=0)  # 读出Excel的第一个sheet

# 数据审查和预处理
set_summary(raw_data)  # 基本状态查看
na_summary(raw_data)  # 缺失值审查
label_summary(raw_data)  # 类样本均衡审查
X = raw_data.drop('response', axis=1)  # 分割X
y = raw_data['response']  # 分割y
X_t = na_replace(X)  # 替换缺失值X
X_t1=type_con(X_t)  #数据类型转换
x_new,enc=symbol_con(X_t1)

select_dimen=dimen_reduce(x_new,y)#降维
print(select_dimen)
print(x_new.shape)
x_new2=x_new[:,select_dimen]
print(x_new2.shape)

# 分类模型训练
model1=AdaBoostClassifier(random_state=0)
model1.fit(x_new2,y)
print(model1.feature_importances_)
print(model1.score(x_new2,y))


#新数据集做预测
test_data = pd.read_excel('./data/order1.xlsx',sheet_name=1)  # 读取要预测的数据集
final_reponse = test_data['final_response']  # 获取最终的目标变量值
test_data = test_data.drop('final_response', axis=1)  # 获得预测的输入变量X
set_summary(test_data)  # 基本状态查看
na_summary(test_data)  # 缺失值审查
test_X_t = na_replace(test_data)  # 替换缺失值
testX_t1=type_con(test_X_t)  #数据类型转换
test_new=symbol_con(testX_t1,enc_object=enc,train=False)

print(test_new.shape)
test_new2=test_new[:,select_dimen]
print(test_new2.shape)
ypre=model1.predict(test_new2)

print('混淆矩阵：')
print(confusion_matrix(final_reponse,ypre))
print('测试集分数：')
print(model1.score(test_new2,final_reponse))
