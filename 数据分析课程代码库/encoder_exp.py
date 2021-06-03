#coding:gbk,
# 编码/one-hot编码
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

le=LabelEncoder()#标签编码
le.fit([1,5,6,100])
print(le.transform([1,1,100,6,5]))
#将离散型数据转换成0到n-1之间的数，n是不同取值的个数
le.fit(['哼','这'])
re=le.transform(['这','哼'])
print(re)


ohe=OneHotEncoder()#独热编码  sparse=True  输出一个matrix稀疏矩阵
ohe.fit([[1],[2],[3],[4],[5],[6],[7]])
re=ohe.transform([[2],[4],[1],[4]]).toarray()
print(re)

ohe=OneHotEncoder()#独热编码
ohe.fit([['Z'],['z']])
re=ohe.transform([['z'],['Z']]).toarray()
print(re)


