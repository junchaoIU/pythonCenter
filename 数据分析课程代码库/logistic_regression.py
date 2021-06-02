#coding:gbk,
#逻辑回归
import pandas as pd
import numpy as np  # numpy库
from sklearn.linear_model import LinearRegression, LogisticRegression  # 批量导入要实现的回归算法

#参数初始化
filename = 'bankloan.xls'
data = pd.read_excel(filename)
x = data.iloc[:,:-1]
y = data.iloc[:,-1]


model_1 =LinearRegression()  # 建立普通线性回归模型对象
model_lr=LogisticRegression(max_iter=1000)
model_dic = [model_1, model_lr]  # 不同回归模型对象的集合
test=x.iloc[3,:].values  #把dataframe转成array

for model in model_dic: 
    model.fit(x,y) 
    print((model.score(x,y)))  
    print(model.predict(np.array(test).reshape(1,-1)))
    

    

