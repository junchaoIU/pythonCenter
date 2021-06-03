#!/usr/bin/env python
#coding:gbk,
import pandas as pd
import numpy as np
s1=pd.Series(range(1,20,5))
s2=pd.Series({'语文':90,'数学':79})
s3=pd.Series([1,2,3],index=['a','b','c'])
print(s1)
print(s2)
print(s3)
s1[3]=-15
s2['语文']=82
print('='*20)

##################时间序列#########################
print(pd.date_range(start='20190601', end='20190630', freq='5D'))
print(pd.date_range(start='20190601', end='20190630', freq='W'))
print(pd.date_range(start='20190601', periods=5, freq='2D'))
print(pd.date_range(start='20190601', periods=8, freq='3H'))
print(pd.date_range(start='201906010300', periods=12, freq='T'))
print('='*20)

###############################################
df = pd.DataFrame(np.random.randint(1, 20, (5,3)),
index=range(5),columns=['A', 'B', 'C'])
print(df)
print('='*20)
df = pd.DataFrame(np.random.randint(5, 15, (13, 3)),
index=pd.date_range(start='201907150900',end='201907152100',
freq='H'),columns=['熟食', '化妆品', '日用品'])
print(df)
print('='*20)
df = pd.DataFrame({'语文':[87,79,67,92],'数学':[93,89,80,77],
'英语':[90,80,70,75]},index=['张三', '李四', '王五', '赵六'])
print(df)
print('='*20)
df = pd.DataFrame({'A':range(5,10), 'B':3})
print(df)
print('='*20)

###############################################
df = pd.read_excel('超市营业额2.xlsx',
usecols=['工号','姓名','时段','交易额'])
print(df[:10])
df2 = pd.read_excel('超市营业额2.xlsx',
skiprows=[1,3,5], index_col=1)
print(df2[:10])

df = pd.read_excel('超市营业额2.xlsx')
print(df[5:11])
print(df.iloc[5,:])
print(df.iloc[[3,5,10]])
print(df.iloc[[3,5,10],[0,1,4]])
print(df[['姓名', '时段', '交易额']][:5])
print(df[:10][['姓名', '日期', '柜台']])
print(df.loc[[3,5,10], ['姓名','交易额']])
print(df[df['交易额']>1700])
print(df['交易额'].sum())
print(df[df['时段']=='14：00-21：00']['交易额'].sum())
print(df[(df.姓名=='张三')&(df.时段=='14：00-21：00')][:10])
print(df[df['柜台']=='日用品']['交易额'].sum())
print(df[df['姓名'].isin(['张三','李四'])]['交易额'].sum())
print(df[df['交易额'].between(800,850)])
#######################################################
print('='*20)
print(df.head())
print(df.describe())
print(df['交易额'].describe())
print(df.nsmallest(3, '交易额'))
print(df.nlargest(5, '交易额'))
print(df['日期'].max())
index = df['交易额'].idxmin()
print(index)
print(df.loc[index,'姓名'])
##########################################
print('='*20)
print(df.sort_values(by=['交易额','工号'], ascending=False)[:12])
print(df.sort_values(by=['交易额','工号'], ascending=[False,True])[:12])

'''d3=pd.read_excel('code3_data.xlsx')#读取Excel，创建Dataframe格式
print(d3.head())
print(d3.describe())

d3data=pd.read_excel('code3_data.xlsx', sheet_name='Sheet1',usecols=[0,1])
print(d3data)

print(d3data['A'])
print(d3data.loc[0])
print(d3data.loc[2][1])
print(d3data.loc[0]['A'])'''

