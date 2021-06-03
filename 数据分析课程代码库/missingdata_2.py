#encoding:utf-8

#拉格朗日插值补全缺失值

import pandas as pd
import numpy as np
from scipy.interpolate import lagrange

inputfile = 'catering_sale.xls' #销量数据路径
#outputfile='../data/sales.xls'
outputfile='sales.xls'

data=pd.read_excel(inputfile)
print(data)
print(data[(data.销量 <400)| (data.销量>5000) ]) #过滤异常值
#pd.set_option('mode.chained_assignment', None)
data['销量'][(data.销量 <400)| (data.销量>5000)]=None #将其变为空值

#自定义列向量插值函数
#s为列向量，n为被插值的位置，k为取前后的数据个数，默认为5
def ployinterp_column(s, n, k=5):
  yindex=list(range(n-k,n))+list(range(n+1, n+1+k))
  y=s[n-k:n+k+1]
  y=y[y.notnull()]
  y1=np.array(y)
  return lagrange(yindex, y1)(n) #插值并返回插值结果
  
print(len(data))
#逐个元素判断是否需要插值
for j in range(len(data)):
	if (np.isnan(data.loc[j]['销量'])): #如果为空即插值
		if j>4 and (len(data)-j)>5:
			data.loc[j,'销量'] = ployinterp_column(data['销量'], j)
		else:
			print('not enough data to do lagrange')

data.to_excel(outputfile,index=0) #输出结果，写入文件
'''

#拉格朗日插值的使用方法
x = np.array([0, 0.7,1, 1.3,2])
y=x**3
print(y)
poly = lagrange(x, y)
print(poly(1.5))
print()'''

