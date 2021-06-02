#coding:gbk,
import os
import pandas as pd
import pyprind
import numpy as np
 
basepath='./data/aclImdb'
labels={'pos':1,'neg':0}


pbar=pyprind.ProgBar(50000)#生成进度条，50000表示更新50000次
df=pd.DataFrame()
for s in ('test','train'):
	for l in ('pos','neg'):
		path=os.path.join(basepath,s,l)#拼接路径
		for file in os.listdir(path):
			with open(os.path.join(path,file),'r',encoding='utf-8') as infile:
				txt=infile.read()
			df=df.append([[txt,labels[l]]],ignore_index=True)
			pbar.update()
df.columns=['review','sentiment']
print(df.head())

##打乱数据,并存储为CSV数据
np.random.seed(0)
df=df.reindex(np.random.permutation(df.index))
df.to_csv('movie_data.csv',index=False,encoding='utf-8')
 
#转成词向量
df=pd.read_csv('movie_data.csv',encoding='utf-8')
df=df[0:500]

from sklearn.feature_extraction.text import CountVectorizer
count=CountVectorizer(stop_words='english',max_df=.1,max_features=5000)
X=count.fit_transform(df['review'].values)
print(X.shape)

#LDA模型

from sklearn.decomposition import LatentDirichletAllocation
lda=LatentDirichletAllocation(n_components=10,
                              random_state=1,learning_method='online')
X_topics=lda.fit_transform(X)
print(lda.components_)

#显示前10个主题中每个主题的5个最重要的单词
n_top_word=5
feature_names=count.get_feature_names()
for topic_idx, topic in enumerate(lda.components_):
	print('Topic %d:', (topic_idx+1))
	print(' '.join([feature_names[i] for i in topic.argsort()[:-n_top_word-1:-1]]))
	


